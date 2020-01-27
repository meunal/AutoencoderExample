import tensorflow as tf
import numpy as np
import mnist
import random
from PIL import Image


class AutoEncoder:
    def __init__(self, x_train, hidden_sizes=[128], num_layers=1, activations=None):
        assert type(hidden_sizes) == list and len(hidden_sizes) == num_layers,\
            'hidden_sizes must be a list and its length should match with num_layers'
        self.x_train = x_train
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.activations = {'encoder': [], 'decoder': []}
        self.variables = {'weights': {}, 'biases': {}}
        self.assign_wb()
        self.assign_activations(activations)

    def assign_activations(self, activations_arr):
        activation_map = {'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh}
        if activations_arr is None:
            for _ in range(self.num_layers):
                self.activations['encoder'].append(tf.nn.sigmoid)
                self.activations['decoder'].append(tf.nn.sigmoid)
        else:
            assert self.num_layers * 2 == len(activations_arr), 'Activations array must have the same length with num_layers * 2'
            for i in range(self.num_layers):
                self.activations['encoder'].append(activation_map[activations_arr[i]])
                self.activations['decoder'].append(activation_map[activations_arr[i + self.num_layers]])

    def assign_wb(self):
        sizes = [self.x_train.shape[1]] + self.hidden_sizes
        r_sizes = [el for el in reversed(sizes)]
        for i in range(self.num_layers):
            self.variables['weights']['encoder_%d' % i] = tf.Variable(tf.truncated_normal([sizes[i], sizes[i + 1]], stddev=0.1))
            self.variables['biases']['encoder_%d' % i] = tf.Variable(tf.truncated_normal([sizes[i + 1]], stddev=0.1))
            self.variables['weights']['decoder_%d' % i] = tf.Variable(tf.truncated_normal([r_sizes[i], r_sizes[i + 1]], stddev=0.1))
            self.variables['biases']['decoder_%d' % i] = tf.Variable(tf.truncated_normal([r_sizes[i + 1]], stddev=0.1))

    def encoder(self, x):
        for i in range(self.num_layers):
            name = 'encoder_%d' % i
            enc = self.activations['encoder'][i](tf.add(tf.matmul(x if i == 0 else enc, self.variables['weights'][name]), self.variables['biases'][name]))
        return enc

    def decoder(self, hidden_rep):
        for i in range(self.num_layers):
            name = 'decoder_%d' % i
            dec = self.activations['decoder'][i](tf.add(tf.matmul(hidden_rep if i == 0 else dec, self.variables['weights'][name]), self.variables['biases'][name]))
        return dec

    def train(self, lr=0.00005, epoch=5, batch_size=50):
        X = tf.placeholder(tf.float32, [None, self.x_train.shape[1]])
        encoder = self.encoder(X)
        decoder = self.decoder(encoder)
        y = X
        y_hat = decoder

        loss = tf.reduce_mean(tf.math.squared_difference(y, y_hat))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        initialize = tf.global_variables_initializer()

        saver = tf.train.Saver()

        session = tf.Session()
        session.run(initialize)

        for e in range(epoch):
            np.random.shuffle(self.x_train)
            for i in range(self.x_train.shape[0] // batch_size):
                _, batch_loss = session.run([optimizer, loss], feed_dict={X: self.x_train[i * batch_size : (i + 1) * batch_size]})
                if i == 0:
                    print("[Epoch %d, Step %d]. Loss -> %.4f" % (e + 1, i + 1, batch_loss))
                if i == (self.x_train.shape[0] // batch_size) - 1 and e == epoch - 1:
                    saver.save(session, 'mnist-autoencoder', e)

    def test(self, im):
        X = tf.placeholder(tf.float32, [None, im.shape[1]])
        encoder = self.encoder(X)
        decoder = self.decoder(encoder)
        y = X
        y_hat = decoder
        saver = tf.train.Saver()
        session = tf.Session()
        saver.restore(session, './mnist-autoencoder-99')
        prediction = session.run([decoder], feed_dict={X: im})
        prediction = np.asarray(prediction).reshape((28, 28)) * 255.
        im = im.reshape((28, 28)) * 255.

        image = Image.fromarray(np.concatenate((im, prediction), axis=0).astype('uint8'), 'L')
        image.save('/Users/mesuterhanunal/Desktop/images/%d.png' % random.randint(0, 100000))


if __name__ == '__main__':
    X_train, _, X_test, Y_test = mnist.load()
    X_train = np.true_divide(X_train, 255.0)
    X_test = np.true_divide(X_test, 255.0)

    autoencoder = AutoEncoder(X_train, hidden_sizes=[64, 64, 2], num_layers=3, activations=['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'relu'])
    # autoencoder.train(lr=0.00002, epoch=100, batch_size=64)

    for i in range(10):
        idx = np.random.choice(np.argwhere(Y_test == i).reshape(-1), 1)
        autoencoder.test(X_test[idx].reshape((1, 784)))
