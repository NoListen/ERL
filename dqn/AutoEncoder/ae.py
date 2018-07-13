import numpy as np
import tensorflow as tf

from .simple_replay_memory import SimpleDataSet
# util is used for AE
from .util import conv2d, linear
from ..base import BaseModel
from functools import reduce
from ..utils import onehot_actions



class AutoEncoder(BaseModel):
    def __init__(self, name, sess, config):
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            super(AutoEncoder, self).__init__(config)
            self._init(sess, config)

    def _init(self,
                 sess,
                 config):
        # At first, I will assume that all the underlying configurations are OK.
        self.sess = sess
        self.features = None
        self.latent_loss = None
        self.generation_loss = None
        self.ae_loss = None
        self.avg_ae_loss = None
        # NHWC default
        self.memory = SimpleDataSet(config, np.random.RandomState(), 'NHWC')
        self.ae_optim = None
        self.gen_loss_list = []
        self.build_network()
        self.sess.run(tf.initialize_all_variables())

    def train(self, t, batch_size):
        # Using train loss as the standard is suitable ?
        gen_loss_list = []
        for i in range(t):
            s_t, s_t_plus_1, action = self.memory.random_batch(batch_size)
            oh_actions = onehot_actions(action)
            _, gen_loss = self.sess.run([self.ae_optim, self.generation_loss],
                                       feed_dict={self.s_t: s_t, self.s_t_plus_1: s_t_plus_1,
                                                  self.action: oh_actions})
            gen_loss_list += list(gen_loss)
        self.avg_ae_loss = np.mean(gen_loss_list)

    def ae_train_mini_batch(self, step, batch_size):
        s_t, s_t_plus_1, action = self.memory.random_batch(batch_size)
        oh_actions = onehot_actions(action)

        _, gen_loss = self.sess.run([self.ae_optim, self.generation_loss],
                                    feed_dict={self.s_t: s_t, self.s_t_plus_1: s_t_plus_1,
                                                  self.action: oh_actions})
        if self.avg_ae_loss:
            self.avg_ae_loss = 0.99 * self.avg_ae_loss + 0.01 * np.mean(gen_loss.flatten())
        else:
            if step >= self.ae_start:
                self.avg_ae_loss = np.mean(gen_loss.flatten())

    def build_network(self):
        self.s_t, self.features = self.build_encoder("encoder")
        self.s_t_plus_1 = tf.placeholder('float32', [None, self.ae_screen_height, self.ae_screen_width],
                                    name='s_t_plus_1')
        s_t_plus_1 = tf.reshape(self.s_t_plus_1, [-1, self.ae_screen_height, self.ae_screen_width, 1])
        self.action, self.z = self.build_latent(self.features, "latent")
        self.g = self.build_deconv(self.z, "decoder")

        # Maybe AE doesn't always to be updated, we can only compute z instead of running the optimizer.
        with tf.variable_scope('ae_optimizer'):
            self.generation_loss = -tf.reduce_sum(s_t_plus_1 * tf.log(self.g+1e-8) +
                                                  (1-s_t_plus_1) * (tf.log(1-self.g+1e-8)),[1,2,3])
            self.ae_loss = self.generation_loss
            self.ae_optim = tf.train.RMSPropOptimizer(
                self.ae_learning_rate, momentum=0.9, decay=0.95, epsilon=1e-4).minimize(self.ae_loss)


    def build_encoder(self, scope="prediction"):
        activation_fn = tf.nn.relu
        conv_initializer = tf.truncated_normal_initializer(0, 0.02)

        with tf.variable_scope(scope):
            input_ = tf.placeholder('float32', [None, self.ae_screen_height, self.ae_screen_width, self.history_length],
                                    name='s_t')

            l1 = conv2d(input_, kernel_size=[5, 5], stride=[3, 3],
                        output_channel=32, name="l1",
                        initializer=conv_initializer,
                        activation_fn=activation_fn,
                        data_format="NHWC",
                        padding="SAME", weight_return=False)

            l2 = conv2d(l1, kernel_size=[3, 3], stride=[2, 2],
                        output_channel=64, name="l2",
                        initializer=conv_initializer,
                        activation_fn=activation_fn,
                        data_format="NHWC",
                        padding="SAME", weight_return=False)

            shape = l2.get_shape().as_list()
            l2_flat = tf.reshape(l2, [-1, reduce(lambda x, y: x * y, shape[1:])])

            return input_, l2_flat

    def build_latent(self, input_, scope):
        activation_fn = tf.nn.relu
        dense_initializer = tf.random_normal_initializer(stddev=0.02)

        with tf.variable_scope(scope):
            action = tf.placeholder('float32', [None, self.n_action], name="action")
            guessed_z = linear(input_, self.latent_size, "guessed_z", initializer = dense_initializer,
                               activation_fn=activation_fn, weight_return = False)
            guessed_z = tf.concat([guessed_z, action], axis = 1)
        return action, guessed_z

    def build_deconv(self, z, scope):
        activation_fn = tf.nn.relu
        initializer = tf.truncated_normal_initializer(stddev=0.02)
        dense_initializer = tf.random_normal_initializer(stddev=0.02)

        with tf.variable_scope(scope):
            # Maybe change it to VAE.
            z_develop = linear(z, 64*7*7, "z_matrix", initializer=dense_initializer, activation_fn=activation_fn,
                               weight_return=False) # 64*7*7

            z_matrix = tf.reshape(z_develop, [-1, 7, 7, 64])
            d1 = tf.contrib.layers.convolution2d_transpose(z_matrix, 32, [3, 3], [2, 2], padding="SAME",
                                                           data_format="NHWC", activation_fn=activation_fn,
                                                           weights_initializer=initializer,
                                                           biases_initializer=tf.constant_initializer(0.0), scope="d1")

            # Predict only one frame.
            d2 = tf.contrib.layers.convolution2d_transpose(d1, 1, [5, 5], [3, 3], padding="SAME",
                                                           data_format="NHWC", activation_fn=tf.nn.sigmoid,
                                                           weights_initializer=initializer,
                                                           biases_initializer=tf.constant_initializer(0.0), scope="d2")

            print("The deconv's output is", d2.get_shape().as_list())

        return d2

    # It could happen when select data and re-evaluate data-batch
    def evaluate_sample(self, s_t_plus_1, oh_action):
        if not self.avg_ae_loss:
            return np.zeros(s_t_plus_1.shape[0])
        gen_loss = self.sess.run([self.generation_loss], feed_dict={self.s_t: self.memory.last_phi()[None],
                                                                       self.s_t_plus_1: s_t_plus_1,
                                                                       self.action: oh_action})

        return np.maximum(gen_loss/self.avg_ae_loss-1, 0)

    def predict(self):
        s_t = self.memory.last_phi()
        action = self.memory.last_action()
        oh_action = onehot_actions([action])
        g = self.sess.run(self.g, feed_dict={self.s_t: s_t[None], self.action: oh_action })
        return g

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
