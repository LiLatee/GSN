import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import scipy.misc
import os


def preprocess_images(images):
    images = images.reshape((images.shape[0], 784)) / 255.
    return images.astype('float32')
    # return np.where(images > .5, 1.0, 0.0).astype('float32')


# fully-conected layer
class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]),
            name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


# merge images
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image
    return img

# save image on local machine


def ims(name, img):
    # print img[:10][:10]
    scipy.misc.toimage(img, cmin=0, cmax=1).save(name)


class Model(tf.Module):
    def __init__(self, name=None):
        super(Model, self).__init__(name=name)
        # First we download the MNIST dataset into our local machine.
        # self.mnist = tfds.load("binarized_mnist", data_dir="out/" )['train']
        train_size = 60000
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
        self.mnist = tf.data.Dataset.from_tensor_slices(
            preprocess_images(train_images)).shuffle(train_size)
        # print(list(self.mnist.as_numpy_iterator())[0])

        print("------------------------------------")
        print("MNIST Dataset Succesufully Imported")
        print("------------------------------------")
        self.n_samples = (len(list(self.mnist)))

        # We set up the model parameters
        # ------------------------------
        # image width,height
        self.img_size = 28
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
        # workaround for variable_scope(reuse=True)

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

    @tf.function
    def read_basic(self, x, x_hat, h_dec_prev):
        return tf.concat([x, x_hat], 1)

    # encoder function for attention patch
    @tf.function
    def encode(self, prev_state, image):
        # update the RNN with our image
        # with tf.variable_scope("encoder",reuse=self.share_parameters):
        hidden_layer, next_state = self.lstm_enc(image, prev_state)

        # map the RNN hidden state to latent variables
        # with tf.variable_scope("mu", reuse=self.share_parameters):
        mu = self.d1(hidden_layer)
        # with tf.variable_scope("sigma", reuse=self.share_parameters):
        logsigma = self.d2(hidden_layer)
        sigma = tf.exp(logsigma)

        return mu, logsigma, sigma, next_state

    @tf.function
    def sampleQ(self, mu, sigma):
        return mu + sigma*self.e

    # decoder function
    @tf.function
    def decode_layer(self, prev_state, latent):
        # update decoder RNN using our latent variable
        # with tf.variable_scope("decoder", reuse=self.share_parameters):
        hidden_layer, next_state = self.lstm_dec(latent, prev_state)

        return hidden_layer, next_state

    # write operation without attention
    @tf.function
    def write_basic(self, hidden_layer):
        # map RNN hidden state to image
        # with tf.variable_scope("write", reuse=self.share_parameters):
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
            print(r.get_shape())
            # encode to guass distribution
            self.mu[t], self.logsigma[t], self.sigma[t], self.enc_state = self.encode(
                self.enc_state, tf.concat([r, self.h_dec_prev], 1))
            # sample from the distribution to get z
            self.z = self.sampleQ(self.mu[t], self.sigma[t])
            # sanity check
            print(self.z.get_shape())
            # retrieve the hidden layer of RNN
            self.h_dec, self.dec_state = self.decode_layer(self.dec_state, self.z)
            # sanity check
            print(self.h_dec.get_shape())
            # map from hidden layer
            self.cs[t] = c_prev + self.write_basic(self.h_dec)
            self.h_dec_prev = self.h_dec

        # # Loss function
        # self.generated_images = tf.nn.sigmoid(self.cs[-1])
        # self.generation_loss = tf.reduce_mean(-tf.reduce_sum(x * tf.math.log(
        #     1e-10 + self.generated_images) + (1-x) * tf.math.log(1e-10 + 1 - self.generated_images), 1))

        # kl_terms = [0]*self.sequence_length
        # for t in range(self.sequence_length):
        #     mu2 = tf.square(self.mu[t])
        #     sigma2 = tf.square(self.sigma[t])
        #     logsigma = self.logsigma[t]
        #     # each kl term is (1xminibatch)
        #     kl_terms[t] = 0.5 * \
        #         tf.reduce_sum(mu2 + sigma2 - 2*logsigma, 1) - \
        #         self.sequence_length*0.5
        # self.latent_loss = tf.reduce_mean(tf.add_n(kl_terms))
        # self.cost = self.generation_loss + self.latent_loss

        # Optimization
        # optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
        # optimizer = tf.keras.optimizers.Adam(1e-4)

        # with tf.GradientTape() as tape:
        #     # grads = optimizer.compute_gradients(self.cost)
        #     grads = tape.gradient(self.cost)
        #     # grads = optimizer.compute_gradients(self.cost)
        #     for i, (g, v) in enumerate(grads):
        #         if g is not None:
        #             grads[i] = (tf.clip_by_norm(g, 5), v)
        #     self.train_op = optimizer.apply_gradients(grads)

        # return self.cs, self.generation_loss, self.latent_loss, self.train_op
        
        return self.cs


def loss(model, x):
    # Loss function
    # model.cs[-1] = model.cs[-1]/100
    generated_images = tf.nn.sigmoid(model.cs[-1])
    # print(tf.reduce_sum(x))
    # print(model.cs[-1])
    print(model.z)
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


def train(model):
    xtrain = model.mnist.batch(model.batch_size)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    for i in range(20000):
        for x in xtrain:
            with tf.GradientTape() as t:
                cs = model(x)
                gen_loss, lat_loss, cost = loss(model, x)

            grads = t.gradient(cost, model.trainable_variables)
            # print(grads)
            # print(type(grads[0]))
            grads = [tf.clip_by_norm(g, 5) for g in grads if g is not None]
            # for i, g in enumerate(grads):
            #     if g is not None:
            #         # grads[i] = (tf.clip_by_norm(g,5),v)
            #         grads[i] = tf.clip_by_norm(g, 5)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print("iter %d genloss %f latloss %f" %
                    (i, gen_loss, lat_loss))
            if i % 500 == 0:

                # x_recons=sigmoid(canvas)
                cs = 1.0/(1.0+np.exp(-np.array(cs)))

                for cs_iter in range(10):
                    results = cs[cs_iter]
                    results_square = np.reshape(results, [-1, 28, 28])
                    print(results_square.shape)
                    # ims("results/"+str(i)+"-step-"+str(cs_iter) +
                    #     ".jpg", merge(results_square, [8, 8]))


m = Model()
# print(m.trainable_variables)
# print(m.variables)
train(m)
