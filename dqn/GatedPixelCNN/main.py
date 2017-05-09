import logging

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")
import os
import time

import numpy as np
import tensorflow as tf
import gym
import core.data.cifar_data as cifar
import core.data.mnist_data as mnist
from network import Network
from statistic import Statistic
import utils as util
from tqdm import tqdm

flags = tf.app.flags
from skimage.transform import resize

# network
flags.DEFINE_integer("batch_size", 100, "size of a batch")
flags.DEFINE_integer("gated_conv_num_layers", 2, "the number of gated conv layers")
flags.DEFINE_integer("gated_conv_num_feature_maps", 16,
                     "the number of input / output feature maps in gated conv layers")
flags.DEFINE_integer("output_conv_num_feature_maps", 64, "the number of output feature maps in output conv layers")
flags.DEFINE_integer("q_levels", 8, "the number of quantization levels in the output")
# 4 used in mnist?
# training
flags.DEFINE_float("max_epoch", 100000, "maximum # of epochs")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("grad_clip", 1, "value of gradient to be used for clipping")

# data
flags.DEFINE_string("data", "mnist", "name of dataset [mnist, color-mnist, cifar]")
flags.DEFINE_string("runtime_base_dir", "./", "path of base directory for checkpoints, data_dir, logs and sample_dir")
flags.DEFINE_string("data_dir", "data", "name of data directory")
flags.DEFINE_string("sample_dir", "samples", "name of sample directory")

# generation
flags.DEFINE_string("occlude_start_row", 21, "image row to start occlusion")
flags.DEFINE_string("num_generated_images", 9, "number of images to generate")

# Debug
flags.DEFINE_boolean("is_train", True, "training or testing")
flags.DEFINE_string("log_level", "INFO", "log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
flags.DEFINE_integer("random_seed", 123, "random seed for python")

conf = flags.FLAGS

# logging
logger = logging.getLogger()
logger.setLevel(conf.log_level)

# random seed
tf.set_random_seed(conf.random_seed)
np.random.seed(conf.random_seed)


# preprocess the data into 0-1
def preprocess(q_levels):
    def preprocess_fcn(images):
        # Create the target pixels from the image. Quantize the scalar pixel values into q_level indices.
        target_pixels = np.clip(((images * q_levels).astype('int64')), 0, q_levels - 1)  # [N,H,W,C]
        return (images, target_pixels)

    return preprocess_fcn

def rgb2y(image):
    assert (len(image.shape) == 3)
    assert (image.shape[-1] == 3)
    im_y = np.dot(image[..., :3], [0.229, 0.587, 0.144])
    im_y = resize(im_y, (84, 84), order=1)
    im_y = resize(im_y, (42, 42), order=1)
    im_y = im_y / 255.
    return im_y.astype(np.float32)


def collect_samples(batch_size, env, action_n, ob_shape=(42, 42)):
    samples = []
    # temporally use random policy
    for i in range(batch_size):
        action = np.random.randint(action_n)
        s, r, terminal, _ = env.step(action)
        if terminal:
            env.reset()
        s = rgb2y(s)
        samples.append(s)
        # temporally ignore reward
    samples = np.array(samples).reshape((batch_size,) + ob_shape + (1,))
    q_fun = preprocess(8)
    return q_fun(samples)

# I would find the value range of the image.
def process_density_images(image):
    # image = image / 255.
    density_images = resize(image, (42, 42), order=1)
    return density_images,astype(np.float32)

def process_density_input(samples):
    # NHWC thx!
    q_func = preprocess(8)
    return q_func(samples)



def generate_from_occluded(network, images):
    occlude_start_row = conf.occlude_start_row
    num_generated_images = conf.num_generated_images

    samples = network.generate_from_occluded(images, num_generated_images, occlude_start_row)

    occluded = np.copy(images[0:num_generated_images, :, :, :])
    # render white line in occlusion start row
    # occluded[:, occlude_start_row, :, :] = 255
    return samples, occluded


def train(env, network, stat, sample_dir):
    initial_step = stat.get_t()
    logger.info("Training starts on epoch {}".format(initial_step))

    train_step_per_epoch = 100
    test_step_per_epoch = 10
    action_n = env.action_space.n
    for epoch in range(initial_step, conf.max_epoch):
        start_time = time.time()

        # 1. train
        total_train_costs = []
        for _ in tqdm(xrange(train_step_per_epoch)):
            images = collect_samples(conf.batch_size, env, action_n)
            cost = network.test(images, with_update=True)
            total_train_costs.append(cost)

        # 2. test
        total_test_costs = []
        for _ in tqdm(xrange(test_step_per_epoch)):
            images = collect_samples(conf.batch_size, env, action_n)
            cost = network.test(images, with_update=False)
            total_test_costs.append(cost)

        avg_train_cost, avg_test_cost = np.mean(total_train_costs), np.mean(total_test_costs)
        stat.on_step(avg_train_cost, avg_test_cost)

        # 3. generate samples
        images, _ = collect_samples(conf.batch_size, env, action_n)
        samples, occluded = generate_from_occluded(network, images)
        util.save_images(np.concatenate((occluded, samples), axis=2),
                         42, 42 * 2, conf.num_generated_images, 1,
                         directory=sample_dir, prefix="epoch_%s" % epoch)

        logger.info("Epoch {}: {:.2f} seconds, avg train cost: {:.3f}, avg test cost: {:.3f}"
                    .format(epoch, (time.time() - start_time), avg_train_cost, avg_test_cost))

def get_network():
    util.preprocess_conf(conf)
    network = Network(conf, 42, 42, 1)
    return network

