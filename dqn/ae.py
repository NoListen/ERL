import tensorflow as tf
# util is used for AE
from util import conv2d, linear
from scipy.misc import imsave
import time
import numpy as np
from tqdm import tqdm
from base import BaseModel
from simple_replay_memory import SimpleReplayMemory
import random

# default NHWC
# input is resized to 42x42x1
# I will still like to resize it into 42*42.

class VAE(BaseModel):
    def __init__(self,
                 sess,
                 config):
        # At first, I will assume that all the underlying configurations are OK.
        super(VAE, self).__init__(config)
        self.sess = sess
        # self.dataset = dataset
        self.features = None
        self.latent_loss = None
        self.generation_loss = None
        self.vae_loss = None
        self.avg_vae_loss = None
        self.memory = SimpleReplayMemory(config)

        # self.batch_size = batch_size
        # self.z_mean = None
        # self.z_stddev = None
        self.vae_optim = None
        self.gen_loss_list = []
        self.build_network()
        self.sess.run(tf.initialize_all_variables())

        # # this part is used to save the checkpoints and restart from the same step
        # with tf.variable_scope("step"):
        #     self.step_op = tf.Variable(0, trainable=False, name="step")
        #     self.step_input = tf.placeholder('int32', None, name="step_input")
        #     self.step_assign_op = self.step_op.assign(self.step_input)

    def train(self, samples_generator):
        # Using train loss as the standard is suitable ?
        gen_loss_list = []
        tmp = samples_generator.next()
        for batch in samples_generator:
            batch = batch/255.
            _, gen_loss = self.sess.run([self.vae_optim, self.generation_loss],
                                       feed_dict={self.data: batch})
            gen_loss_list += list(gen_loss)
        from scipy.misc import imsave
        import time
        g = self.sess.run(self.g, feed_dict={self.data: tmp})
        for i in range(g.shape[0]):
            imsave(time.strftime("./fuck/%d%H%M%S")+"%i.jpg"% i, g[i].reshape(42,42))
        # dynamic could be better.
        self.avg_vae_loss = np.mean(gen_loss_list)
        print "update the avg_ae_loss",self.avg_vae_loss
        # if dynamic
        # if self.avg_vae_loss:
        #   self.avg_vae_loss = 0.98 * self.avg_vae_loss + 0.02 * np.mean(gen_loss_list)
        # else:
        #   self.avg_vae_loss = np.mean(self.avg_vae_loss)

    def ae_train_mini_batch(self, batch, step):
        # batch = batch/255.
        _, gen_loss = self.sess.run([self.vae_optim, self.generation_loss],
                                    feed_dict={self.data: batch})
        if self.avg_vae_loss:
            self.avg_vae_loss = 0.99 * self.avg_vae_loss + 0.01 * np.mean(gen_loss.flatten())
        else:
            if step >= self.ae_start:
                self.avg_vae_loss = np.mean(gen_loss.flatten())
        # if self.avg_vae_loss:
        #     print self.avg_vae_loss

    def build_network(self):
        self.data, self.features = self.build_encoder("encoder")
        self.z = self.build_latent(self.features, "latent")
        self.g = self.build_deconv(self.z, "decoder")


        # Maybe AE doesn't always to be updated, we can only compute z instead of running the optimizer.
        with tf.variable_scope('vae_optimizer'):
            self.generation_loss = -tf.reduce_sum(self.data/self.img_scale * tf.log(self.g+1e-8) +
                                                  (1-self.data/self.img_scale) * (tf.log(1-self.g+1e-8)),[1,2,3])
            # self.latent_loss = tf.reduce_sum(0.5 * (tf.square(self.z_mean) + tf.square(self.z_stddev) -
            #                                         2.0 * tf.log(self.z_stddev+self.epsilon) - 1.0),1)
            self.vae_loss = self.generation_loss
            # self.vae_loss = self.generation_loss + self.latent_loss
            # self.vae_optim = tf.train.AdamOptimizer(self.vae_learning_rate).minimize(self.vae_loss)
            self.vae_optim = tf.train.RMSPropOptimizer(
                self.vae_learning_rate, momentum=0.9, decay=0.95, epsilon=1e-4).minimize(self.vae_loss)


    def build_encoder(self, scope="prediction"):
        activation_fn = tf.nn.relu
        conv_initializer = tf.truncated_normal_initializer(0, 0.02)
        dense_initializer = tf.random_normal_initializer(stddev=0.02)

        with tf.variable_scope(scope):
            input_ = tf.placeholder('float32', [None, self.ae_screen_height, self.ae_screen_width, 1],
                                    name='s')
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

            # l3, w_collection["l3_w"], w_collection["l3_b"] = conv2d(l2, kernel_size=[3, 3], stride=[1, 1],
            #                                                         output_channel=64, name="l3",
            #                                                         initializer=conv_initializer,
            #                                                         activation_fn=activation_fn,
            #                                                         data_format=self.dataset.data_format,
            #                                                         padding="VALID", weight_return=True)

            shape = l2.get_shape().as_list()
            l2_flat = tf.reshape(l2, [-1, reduce(lambda x, y: x * y, shape[1:])])

            return input_, l2_flat

    def build_latent(self, input_, scope):
        activation_fn = tf.nn.relu
        dense_initializer = tf.random_normal_initializer(stddev=0.02)

        with tf.variable_scope(scope):
            # self.z_mean = linear(input_, self.latent_size, "z_mean", initializer = dense_initializer,
            #                      activation_fn=activation_fn, weight_return = False)
            # self.z_stddev = linear(input_, self.latent_size, "z_stddev", initializer = dense_initializer,
            #                        activation_fn=activation_fn, weight_return = False)
            # nsamples = self.z_mean.get_shape().as_list()[0]
            # samples = tf.random_normal([nsamples, self.latent_size], 0, 1, dtype=tf.float32)
            # guessed_z = self.z_mean + (self.z_stddev * samples)
            guessed_z = linear(input_, self.latent_size, "guessed_z", initializer = dense_initializer,
                               activation_fn=activation_fn, weight_return = False)
        return guessed_z

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

            # d1 = deconv2d(z_matrix, [3, 3], [1, 1], [self.batch_size, 9, 9, 64], "d1", initializer=initializer,
            #               activation_fn=activation_fn, data_format=self.dataset.data_format, padding="VALID")

            d2 = tf.contrib.layers.convolution2d_transpose(d1, 1, [5, 5], [3, 3], padding="SAME",
                                                           data_format="NHWC", activation_fn=tf.nn.sigmoid,
                                                           weights_initializer=initializer,
                                                           biases_initializer=tf.constant_initializer(0.0), scope="d2")
            #
            # d3 = tf.contrib.layers.convolution2d_transpose(d2, 1, [4, 4], [2, 2], padding="VALID",
            #                                                data_format="NHWC", activation_fn=tf.nn.sigmoid,
            #                                                weights_initializer=initializer,
            #                                                biases_initializer=tf.constant_initializer(0.0), scope="d3")
            print d2.get_shape().as_list(), "d2"

        return d2

    # It could happen when select data and re-evaluate data-batch
    def evaluate_sample(self, data, genre="train"):
        if not self.avg_vae_loss:
            return np.zeros(data.shape[0])
        gen_loss, g = self.sess.run([self.generation_loss, self.g], feed_dict={self.data: data})

        # print np.max(g), np.max(data/255.)
        # if genre == "ep":
        for i in range(data.shape[0]):
            imsave("./fuck/"+time.strftime("%d%H%M%S") + "_%i_%f.jpg" % (i, gen_loss[i]), g[i].reshape(42,42))
            imsave("./fuck/"+time.strftime("%d%H%M%S") + "_%i_%f_o.jpg" % (i, gen_loss[i]), data[i].reshape(42,42)/255.)

        return np.maximum(gen_loss/self.avg_vae_loss-1, 0)
        # return np.maximum(gen_loss/self.avg_vae_loss-1-self.ae_threshold, 0)
