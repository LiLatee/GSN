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

results_dir = 'results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

class Memory:
    size = 0
    images = np.array([])
    batch_size = 0

    def __init__(self, batch_size=64, img_size=64):
        self.img_size = img_size
        self.size = 0
        self.batch_size = batch_size

    def load_images_from_folder(self, folder, max_img_counnter):
        images = []
        img_c = 0
        for filename in os.listdir(folder):
            img = skimage.io.imread(os.path.join(folder, filename))
            if img is not None:
                gray_image = rgb2gray(img)
                image_resized = resize(gray_image, (self.img_size, self.img_size),
                                       anti_aliasing=True)
#                 imgplot = plt.imshow(gray_image)
#                 plt.show()
#                 exit()
                data = np.array(image_resized)
                flattened = data.flatten()
                images.append(flattened)
                img_c += 1
            if (img_c > max_img_counnter-1):
                self.size = self.size+img_c
                self.images = np.array(images).astype(np.float32)
                return "DONE"

        self.size = self.size+img_c
        self.images = np.array(images)
        return "DONE"  # "self.images"


# merge images
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w] = image
    return img

# save image on local machine


def ims(name, img):
    # print img[:10][:10]
    im = Image.fromarray(img*255)
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
    def __init__(self, name=None):
        super(Model, self).__init__(name=name)

        # We set up the model parameters
        # ------------------------------
        # image width,height
        self.img_size = 64
        # read glimpse grid width/height
        self.attention_n = 5
        # number of hidden units / output size in LSTM
        self.n_hidden = 256
        # QSampler output size
        self.n_z = 10
        # MNIST generation sequence length
        self.sequence_length = 10
        # training minibatch size
        self.batch_size = 64

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
        self.d3 = Dense(self.n_hidden, self.img_size**2)
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
            c_prev = tf.zeros((self.batch_size, self.img_size**2)
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
    # model.cs[-1] = model.cs[-1]/100
    generated_images = tf.nn.sigmoid(model.cs[-1])
    # print(tf.reduce_sum(x))
    # print(model.cs[-1])
    # print(model.z)
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
def train(model, xtrain, optimizer, i):
    with tf.GradientTape() as t:
        cs = model(x)
        gen_loss, lat_loss, cost = loss(model, x)

    grads = t.gradient(cost, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 5) for g in grads if g is not None]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print(f"genloss {gen_loss} latloss {lat_loss}")
    gl.append(gen_loss)
    ll.append(lat_loss)
    if i % 5000 == 0:
        cs = 1.0/(1.0+np.exp(-np.array(cs)))  # x_recons=sigmoid(canvas)
        print("ZAPIS")
        try:
            tf.saved_model.save(model, results_dir + 'models/model-'+str(i))
            save_losses(gl, ll, i)
        except Exception as e:
            print("błąd zapisu: ", str(e))
        for cs_iter in range(10):
            results = cs[cs_iter]
            results_square = np.reshape(results, [-1, 64, 64])
            print(results_square.shape)
            img_to_save = merge(results_square, [8, 8])
            ims(results_dir+"images/"+str(i)+"-step-"+str(cs_iter) + ".jpeg", img_to_save)

def save_losses(gl, ll, i):
    losses_dir = results_dir + "losses"
    if not os.path.exists(losses_dir):
        os.makedirs(losses_dir)
    out_file = os.path.join(losses_dir, "draw_data-"+str(i)+".npy")
    np.save(out_file, [gl, ll])
    print("Outputs saved in file: %s" % out_file)


imgs_path = "./out_aug_64x64"
mem = Memory(img_size=64)
mem.load_images_from_folder(imgs_path, 90000)
print(type(mem.images))
print(len(mem.images))
print((mem.images[0].shape))

data = tf.data.Dataset.from_tensor_slices(mem.images).shuffle(len(mem.images)).batch(batch_size=64, drop_remainder=True)
m = Model()
optimizer = tf.keras.optimizers.Adam(1e-4)
c = 0
for i in range(100):
    for j, x in enumerate(data):
        print('iter ', c)
        try:
            train(m, x, optimizer, c)
        except Exception as e:
            print("BLAD: ", e)
        c += 1
