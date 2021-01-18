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
    shape = (h * size[0], w * size[1], 3) if len(result_image_shape) == 3 else (h * size[0], w * size[1])
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
    # print(img)
    # print(img.shape)
    # print(img.dtype)
    img = img.astype(np.uint8)
    print(img)
    print(img.shape)
    print(img.dtype)
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
    def __init__(self, img_flattened_size, name=None, batch_size=64, ):
        super(Model, self).__init__(name=name)

        # We set up the model parameters
        # ------------------------------
        # image width,height
        # self.img_size = 64
        self.img_flattened_size = img_flattened_size
        # read glimpse grid width/height
        self.attention_n = 5
        # number of hidden units / output size in LSTM
        self.n_hidden = 256
        # QSampler output size
        self.n_z = 10
        # generation sequence length
        self.sequence_length = 10
        # training minibatch size
        self.batch_size = batch_size

        # self.data = tf.data.Dataset.from_tensor_slices(data).batch(
        #     batch_size=self.batch_size, drop_remainder=True).shuffle(len(data))
        # # First we download the MNIST dataset into our local machine.
        # train_size = 60000
        # (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
        # # self.mnist = tfds.load("mnist")['train']

        # self.mnist = tf.data.Dataset.from_tensor_slices(
        #     preprocess_images(train_images)).shuffle(train_size)

        # print("------------------------------------")
        # print("MNIST Dataset Succesufully Imported")
        # print("------------------------------------")

        # Build our model
        self.e = tf.random.normal(
            (self.batch_size, self.n_z), mean=0, stddev=1)  # Qsampler noise
        self.lstm_enc = tf.keras.layers.LSTMCell(self.n_hidden)
        self.lstm_dec = tf.keras.layers.LSTMCell(self.n_hidden)
        self.d1 = Dense(self.n_hidden, self.n_z)
        self.d2 = Dense(self.n_hidden, self.n_z)
        self.d3 = Dense(self.n_hidden, self.img_flattened_size)
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

    def __call__(self, x):
        # Construct the unrolled computational graph
        for t in range(self.sequence_length):
            # error image + original image
            c_prev = tf.zeros((self.batch_size, self.img_flattened_size)
                              ) if t == 0 else self.cs[t-1]
            x_hat = x - tf.sigmoid(c_prev)
            # read the image
            r = self.read_basic(x, x_hat, self.h_dec_prev)
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
            self.cs[t] = c_prev + self.write_basic(self.h_dec)
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
        cs = 1.0/(1.0+np.exp(-np.array(cs)))  # x_recons=sigmoid(canvas)
        print("ZAPIS")
        try:
            # model.save(results_dir + 'modelsRGB/model-'+str(i))
            # tf.saved_model.save(model, results_dir + 'modelsRGB/model-'+str(i))
            joblib.dump(model, os.path.join(dirs_dict["models"], 'model-'+str(i)))
            save_losses(gl, ll, i, dir=dirs_dict["losses"])
        except Exception as e:
            print("błąd zapisu: ", str(e))
        for cs_iter in range(sequence_length):
            results = cs[cs_iter]
            results_square = np.reshape(results, [-1] + result_image_shape)
            print(results_square.shape)
            img_to_save = merge(results_square, [8, 8], result_image_shape)

            img_save_path = os.path.join(dirs_dict["images"], str(i)+"-step-" + str(cs_iter) + ".jpeg" )
            ims(img_save_path, img_to_save)


def save_losses(gl, ll, i, dir):
    out_file = os.path.join(dir, "draw_data-"+str(i)+".npy")
    np.save(out_file, [gl, ll])
    print("Outputs saved in file: %s" % out_file)


def load_images_from_dir(dir, number, skip=0, gray=False):
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
            image_resized = resize(img, (64, 64, 1), anti_aliasing=True)
        else:    
            image_resized = resize(img, (64, 64, 3), anti_aliasing=True)

        data = np.array(image_resized)
        flattened = data.flatten()
        images.append(flattened)

        ctr += 1
        if ctr == number:
            images = np.array(images).astype(np.float32)
            return images
        

if __name__ == "__main__":
    ## Utworzenie katalogów.
    date = datetime.now().strftime("%d-%m-%Y %H:%M")
    results_dir = f"results-{date}/"
    images_dir = results_dir + 'images'
    models_dir = results_dir + 'models'
    losses_dir = results_dir + "losses"
    needed_dirs = [results_dir, images_dir, models_dir, losses_dir]

    for dir in needed_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    dirs_dict ={
        "results": results_dir,
        "images": images_dir,
        "models": models_dir,
        "losses": losses_dir
    }

    ## Załadowanie zdjęć.
    imgs_path = "./cats_heads_64x64"
    is_gray_img = False
    images = load_images_from_dir(dir=imgs_path, number=1000, skip=0, gray=is_gray_img)
    result_image_shape = [64, 64] if is_gray_img else [64, 64, 3]
    ## Uczenie.
    data = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(batch_size=64, drop_remainder=True)
    model = Model(img_flattened_size=images[0].shape[0], batch_size=64)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    c = 0
    for i in range(100):
        for j, x in enumerate(data):
            print('iter ', c)
            try:
                train(
                    model, 
                    x, 
                    optimizer, 
                    c, 
                    result_image_shape=result_image_shape, 
                    dirs_dict=dirs_dict, 
                    sequence_length=model.sequence_length,
                    number_of_steps_to_make_save=5000)
            except Exception as e:
                print("BLĄD: ", e)
            c += 1






### Ładowanie zdjęć za pomocą generatora. Jednak pojedynczy krok uczenia trwa sporo dłużej na tym.

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
