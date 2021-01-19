from PIL import Image
import tensorflow_datasets as tfds
import skimage.io
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
import scipy.misc
import numpy as np
import os
import time
import random
import glob
import joblib
from datetime import datetime


# merge images
def merge(images, size, result_image_shape):
    h, w = images.shape[1], images.shape[2]
    shape = (h * size[0], w * size[1],
             3) if len(result_image_shape) == 3 else (h * size[0], w * size[1])
    img = np.zeros(shape)
    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w] = image
    return img

# save image on local machine


def ims(name, img):
    # print img[:10][:10]

    img = img*255
    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    im = im.convert('RGB')
    im.save(fp=name)
    # scipy.misc.toimage(img, cmin=0, cmax=1).save(name)


def normalize_img(image):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.


def preprocess_images(images):
    images = images.reshape((images.shape[0], 784)) / 255.
    return images.astype('float32')
    # return np.where(images > .5, 1.0, 0.0).astype('float32')


# fully-conected layer
class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features], stddev=0.02),
            name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return y  # tf.nn.relu(y)


class Model(tf.Module):
    def __init__(self, img_shape, attention_n, n_hidden, n_z, sequence_length, name=None, batch_size=64, with_attention=False):
        super(Model, self).__init__(name=name)

        # We set up the model parameters
        # ------------------------------
        assert img_shape[0] == img_shape[1]
        self.img_size = img_shape[0]
        self.img_channels = img_shape[2] if len(img_shape) == 3 else 1
        self.img_flattened_size = self.img_size * self.img_size * self.img_channels
        self.attention_n = attention_n
        self.n_hidden = n_hidden
        self.n_z = n_z
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.with_attention = with_attention

        # Build our model
        self.e = tf.random.normal(
            (self.batch_size, self.n_z), mean=0, stddev=1)  # Qsampler noise
        self.lstm_enc = tf.keras.layers.LSTMCell(self.n_hidden)
        self.lstm_dec = tf.keras.layers.LSTMCell(self.n_hidden)
        self.d1 = Dense(self.n_hidden, self.n_z)
        self.d2 = Dense(self.n_hidden, self.n_z)
        if with_attention:
            self.dense_readA = Dense(self.n_hidden, self.attention_n)
            self.dense_writeAW = Dense(self.n_hidden, self.attention_n*self.attention_n)
            self.dense_writeA = Dense(self.n_hidden, self.attention_n)

            self.write = self.write_attention
            self.read = self.read_attention
        else:
            self.d3 = Dense(self.n_hidden, self.img_flattened_size)
            self.write = self.write_basic
            self.read = self.read_basic

        # Define our state variables
        self.cs = [0] * self.sequence_length  # sequence of canvases
        self.mu, self.logsigma, self.sigma = [
            0] * self.sequence_length, [0] * self.sequence_length, [0] * self.sequence_length

        # Initial states
        self.h_dec_prev = tf.zeros((self.batch_size, self.n_hidden))
        self.enc_state = self.lstm_enc.get_initial_state(
            batch_size=self.batch_size, dtype=tf.float32)
        self.dec_state = self.lstm_dec.get_initial_state(
            batch_size=self.batch_size, dtype=tf.float32)

    # read operation without attention
    def read_basic(self, x, x_hat, h_dec_prev):
        return tf.concat([x, x_hat], 1)

    # encoder function for attention patch
    def encode(self, prev_state, image):
        # update the RNN with our image
        hidden_layer, next_state = self.lstm_enc(image, prev_state)
        # map the RNN hidden state to latent variables
        mu = self.d1(hidden_layer)
        logsigma = self.d2(hidden_layer)
        sigma = tf.exp(logsigma)

        return mu, logsigma, sigma, next_state

    def sampleQ(self, mu, sigma):
        return mu + sigma*self.e

    # decoder function
    def decode_layer(self, prev_state, latent):
        # update decoder RNN using our latent variable
        hidden_layer, next_state = self.lstm_dec(latent, prev_state)
        return hidden_layer, next_state

    # write operation without attention
    def write_basic(self, hidden_layer):
        # map RNN hidden state to image
        decoded_image_portion = self.d3(hidden_layer)
        return decoded_image_portion

#########################################################################################
## Methods used if attention mechanism is on.
    # locate where to put attention filters on hidden layers
    def attn_window(self, scope, h_dec):
        if scope == "read":
            parameters = self.dense_readA(h_dec)
        elif scope == "write":
            parameters = self.dense_writeA(h_dec)
        # center of 2d gaussian on a scale of -1 to 1
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(
            value=parameters, num_or_size_splits=5, axis=1)

        # move gx/gy to be a scale of -imgsize to +imgsize
        gx = (self.img_size+1)/2 * (gx_ + 1)
        gy = (self.img_size+1)/2 * (gy_ + 1)

        sigma2 = tf.exp(log_sigma2)
        # distance between patches
        delta = (self.img_size - 1) / (self.attention_n-1) * tf.exp(log_delta)
        # returns [Fx, Fy, gamma]
        return self.filterbank(gx, gy, sigma2, delta) + (tf.exp(log_gamma),)

    # Construct patches of gaussian filters
    def filterbank(self, gx, gy, sigma2, delta):
        # 1 x N, look like [[0,1,2,3,4]]
        grid_i = tf.reshape(
            tf.cast(tf.range(self.attention_n), tf.float32), [1, -1])
        # individual patches centers
        mu_x = gx + (grid_i - self.attention_n/2 - 0.5) * delta
        mu_y = gy + (grid_i - self.attention_n/2 - 0.5) * delta
        mu_x = tf.reshape(mu_x, [-1, self.attention_n, 1])
        mu_y = tf.reshape(mu_y, [-1, self.attention_n, 1])
        # 1 x 1 x imgsize, looks like [[[0,1,2,3,4,...,27]]]
        im = tf.reshape(
            tf.cast(tf.range(self.img_size), tf.float32), [1, 1, -1])
        # im2 = tf.reshape(
        #     tf.cast(tf.range(self.img_size), tf.float32), [1, 1, -1])
        # list of gaussian curves for x and y
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(tf.negative(tf.square((im - mu_x) / (2*sigma2))))
        Fy = tf.exp(tf.negative(tf.square((im - mu_y) / (2*sigma2))))
        # normalize area-under-curve
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keepdims=True), 1e-8)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keepdims=True), 1e-8)
        return Fx, Fy

    # read operation with attention
    def read_attention(self, x, x_hat, h_dec_prev):
        Fx, Fy, gamma = self.attn_window("read", h_dec_prev)
        # apply parameters for patch of gaussian filters

        def filter_img(img, Fx, Fy, gamma):
            Fxt = tf.transpose(Fx, perm=[0, 2, 1])
            if self.img_channels == 3:
                img = tf.reshape(img, [-1, self.img_size, self.img_size, 3])
            elif self.img_channels == 1:
                img = tf.reshape(img, [-1, self.img_size, self.img_size])

            # apply the gaussian patches
            glimpse = tf.raw_ops.BatchMatMul(
                x=Fy, y=tf.raw_ops.BatchMatMul(x=img, y=Fxt))
            glimpse = tf.reshape(glimpse, [-1, self.attention_n**2])
            # scale using the gamma parameter
            return glimpse * tf.reshape(gamma, [-1, 1])

        x = filter_img(x, Fx, Fy, gamma)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)
        return tf.concat([x, x_hat], 1)

    # write operation with attention
    def write_attention(self, hidden_layer):
        w = self.dense_writeAW(hidden_layer)
        w = tf.reshape(
            w, [self.batch_size, self.attention_n, self.attention_n])
        Fx, Fy, gamma = self.attn_window("write", hidden_layer)
        Fyt = tf.transpose(Fy, perm=[0, 2, 1])
        wr = tf.linalg.matmul(a=Fyt, b=tf.linalg.matmul(a=w, b=Fx))
        wr = tf.reshape(wr, [self.batch_size, self.img_size*self.img_size])  # KOLOR?
        return wr * tf.reshape(1.0/gamma, [-1, 1])
#########################################################################################

    def __call__(self, x):
        # Construct the unrolled computational graph
        for t in range(self.sequence_length):
            # error image + original image
            c_prev = tf.zeros((self.batch_size, self.img_flattened_size)
                              ) if t == 0 else self.cs[t-1]
            x_hat = x - tf.sigmoid(c_prev)
            # read the image
            # r = self.read_basic(x, x_hat, self.h_dec_prev)
            r = self.read(x, x_hat, self.h_dec_prev)
            # sanity check
            # print(r.get_shape())
            # encode to guass distribution
            self.mu[t], self.logsigma[t], self.sigma[t], self.enc_state = self.encode(
                self.enc_state, tf.concat([r, self.h_dec_prev], 1))
            # sample from the distribution to get z
            self.z = self.sampleQ(self.mu[t], self.sigma[t])
            # sanity check
            # print(self.z.get_shape())
            # retrieve the hidden layer of RNN
            self.h_dec, self.dec_state = self.decode_layer(
                self.dec_state, self.z)
            # sanity check
            # print(self.h_dec.get_shape())
            # map from hidden layer
            # self.cs[t] = c_prev + self.write_basic(self.h_dec)
            self.cs[t] = c_prev + self.write(self.h_dec)
            self.h_dec_prev = self.h_dec
        return self.cs


def loss(model, x):
    # Loss function
    generated_images = tf.nn.sigmoid(model.cs[-1])
    generation_loss = tf.reduce_mean(-tf.reduce_sum(x * tf.math.log(
        1e-10 + generated_images) + (1-x) * tf.math.log(1e-10 + 1 - generated_images), 1))

    kl_terms = [0]*model.sequence_length
    for t in range(model.sequence_length):
        mu2 = tf.square(model.mu[t])
        sigma2 = tf.square(model.sigma[t])
        logsigma = model.logsigma[t]
        # each kl term is (1xminibatch)
        kl_terms[t] = 0.5 * \
            tf.reduce_sum(mu2 + sigma2 - 2*logsigma, 1) - \
            model.sequence_length*0.5
    latent_loss = tf.reduce_mean(tf.add_n(kl_terms))
    cost = generation_loss + latent_loss

    return generation_loss, latent_loss, cost


gl = []
ll = []


def train(model, x, optimizer, i, result_image_shape, dirs_dict, sequence_length, number_of_steps_to_make_save):
    with tf.GradientTape() as t:
        cs = model(x)
        gen_loss, lat_loss, cost = loss(model, x)

    grads = t.gradient(cost, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 5) for g in grads if g is not None]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print(f"genloss {gen_loss} latloss {lat_loss}")
    gl.append(gen_loss)
    ll.append(lat_loss)
    if i % number_of_steps_to_make_save == 0 and i != 0:
        # for cs_iter in range(sequence_length):
        #     np.savetxt(f'tests/przed/{cs_iter}.txt', np.array(cs[cs_iter]), delimiter=',')
        # cs = 1.0/(1.0+np.exp(-np.array(cs)))  # x_recons=sigmoid(canvas)
        cs = tf.math.sigmoid(np.array(cs))
        print("="*100)
        print(cs)
        print("ZAPIS")
        try:
            # model.save(results_dir + 'modelsRGB/model-'+str(i))
            # tf.saved_model.save(model, results_dir + 'modelsRGB/model-'+str(i))
            joblib.dump(model, os.path.join(
                dirs_dict["models"], 'model-'+str(i)))
            save_losses(gl, ll, i, dir=dirs_dict["losses"])
        except Exception as e:
            print("błąd zapisu: ", str(e))
        for cs_iter in range(sequence_length):
            results = cs[cs_iter]
            # np.savetxt(f'tests/po/{cs_iter}.txt', results, delimiter=',')
            results_square = np.reshape(results, [-1] + result_image_shape)
            print(results_square.shape)
            img_to_save = merge(results_square, [8, 8], result_image_shape)

            img_save_path = os.path.join(dirs_dict["images"], str(
                i)+"-step-" + str(cs_iter) + ".png")
            ims(img_save_path, img_to_save)


def save_losses(gl, ll, i, dir):
    out_file = os.path.join(dir, "draw_data-"+str(i)+".npy")
    np.save(out_file, [gl, ll])
    print("Outputs saved in file: %s" % out_file)


def load_images_from_dir(dir, number, skip=0, gray=False, img_size=64):
    images = []
    ctr = 0
    dir = dir + "/*.jpg"
    all_images_paths = glob.glob(dir)
    all_images_paths = all_images_paths[skip:]
    random.shuffle(all_images_paths)
    for filename in all_images_paths:
        img = skimage.io.imread(filename)
        if gray:
            img = rgb2gray(img)
            image_resized = resize(
                img, (img_size, img_size, 1), anti_aliasing=True)
        else:
            image_resized = resize(
                img, (img_size, img_size, 3), anti_aliasing=True)

        data = np.array(image_resized)
        flattened = data.flatten()
        images.append(flattened)

        ctr += 1
        if ctr == number:
            images = np.array(images).astype(np.float32)
            return images, img.shape


if __name__ == "__main__":
    ## Parameters
    learning_rate = 1e-4
    batch_size = 64
    img_size = 64 # img_size = width = height
    is_gray_img = True
    # read glimpse grid width/height
    attention_n = 5
    # number of hidden units / output size in LSTM
    n_hidden = 256
    # QSampler output size
    n_z = 30
    # generation sequence length
    sequence_length = 10
    with_attention = True
    

    result_image_shape = [img_size, img_size] if is_gray_img else [img_size, img_size, 3]

    ## Creating directories.
    date = datetime.now().strftime("%d-%m-%Y %H:%M")
    results_dir = f"results-{date}/"
    images_dir = results_dir + 'images'
    models_dir = results_dir + 'models'
    losses_dir = results_dir + "losses"
    needed_dirs = [results_dir, images_dir, models_dir, losses_dir]

    for dir in needed_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    dirs_dict = {
        "results": results_dir,
        "images": images_dir,
        "models": models_dir,
        "losses": losses_dir
    }

    # Load images.
    imgs_path = "./cats_heads_64x64"
    images, imgs_source_shape = load_images_from_dir(
        dir=imgs_path, number=90000, skip=0, gray=is_gray_img)

    
    
    # Learning.
    data = tf.data.Dataset.from_tensor_slices(images).shuffle(
        len(images)).batch(batch_size=batch_size, drop_remainder=True)

    model = Model(
        img_shape=imgs_source_shape, 
        batch_size=batch_size, 
        n_hidden=n_hidden,
        attention_n=attention_n,
        n_z=n_z,
        sequence_length=sequence_length,
        with_attention=with_attention)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    c = 0
    for i in range(100):
        for j, x in enumerate(data):
            print('iter ', c)
            # try:
            train(
                model,
                x,
                optimizer,
                c,
                result_image_shape=result_image_shape,
                dirs_dict=dirs_dict,
                sequence_length=model.sequence_length,
                number_of_steps_to_make_save=5000)
            # except Exception as e:
            #     print("BLĄD: ", e)
            c += 1


# Ładowanie zdjęć za pomocą generatora. Jednak pojedynczy krok uczenia trwa sporo dłużej na tym.

# imgs_path = "./cats_heads_64x64"
# it = data_generator(images_dir="cats_heads_64x64")

# def data_generator(images_dir):
#     images_dir = images_dir + "/*.jpg"
#     all_images = glob.glob(images_dir)
#     random.shuffle(all_images)
#     for filename in all_images:
#         img = skimage.io.imread(filename)
#         # gray_image = rgb2gray(img)
#         gray_image = img
#         image_resized = resize(gray_image, (64, 64, 3),
#                                 anti_aliasing=True)
#         data = np.array(image_resized)
#         flattened = data.flatten()
#         yield flattened

# def load_batch(batch_size=64):
#     batch = []
#     while len(batch) < batch_size:
#         try:
#             batch.append(next(it))
#         except StopIteration:
#             batch = None
#             return batch
#     return batch

# m = Model()
# # m = joblib.load("results/models/model-30")
# optimizer = tf.keras.optimizers.Adam(1e-4)
# c = 0
# for i in range(100):
#     while True:
#         print('iter ', c)
#         batch = load_batch(batch_size=64)
#         batch = np.array(batch).astype(np.float32)
#         batch = tf.data.Dataset.from_tensor_slices(batch).batch(batch_size=64, drop_remainder=True)
#         if batch is None:
#             break

#         try:
#             for x in batch:
#                 train(m, x, optimizer, c)
#         except Exception as e:
#             print("BLAD: ", e)
#         c += 1


# imgs_path = "./cats_heads_64x64"
# mem = Memory(img_size=64)
# mem.load_images_from_folder(imgs_path, 3000)
# print(type(mem.images))
# print(len(mem.images))
# print((mem.images[0].shape))

# data = tf.data.Dataset.from_tensor_slices(mem.images).shuffle(
#     len(mem.images)).batch(batch_size=64, drop_remainder=True)
# m = Model()
# # m = tf.keras.models.load_model("results/modelsRGB/model-100")
# optimizer = tf.keras.optimizers.Adam(1e-4)
# c = 0
# for i in range(100):
#     for j, x in enumerate(data):
#         print('iter ', c)
#         # try:
#         train(m, x, optimizer, c)
#         # except Exception as e:
#         #     print("BLAD: ", e)
#         c += 1
