import logging

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import numpy as np
import tensorflow as tf
from .network import Network
from .utils import preprocess_conf
flags = tf.app.flags
from skimage.transform import resize

# network
flags.DEFINE_integer("gated_conv_num_layers", 2, "the number of gated conv layers")
flags.DEFINE_integer("gated_conv_num_feature_maps", 16,
                     "the number of input / output feature maps in gated conv layers")
flags.DEFINE_integer("output_conv_num_feature_maps", 64, "the number of output feature maps in output conv layers")
flags.DEFINE_integer("q_levels", 10, "the number of quantization levels in the output")
flags.DEFINE_float("max_epoch", 100000, "maximum # of epochs")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("decay", 0.95, "decay")
flags.DEFINE_float("momentum",0.9, "momentum")
flags.DEFINE_float("epsilon", 1e-4, "epsilon")
flags.DEFINE_float("grad_norm_clip", 40.0, "grad norm of clip")

flags.DEFINE_float("grad_clip", 1, "value of gradient to be used for clipping")

flags.DEFINE_string("log_level", "INFO", "log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

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
        return target_pixels

    return preprocess_fcn

def process_density_images(image):
    return image.reshape(-1, 42, 42, 1).astype(np.float32)

def process_density_input(samples):
    # NHWC thx!
    q_func = preprocess(conf.q_levels)
    return q_func(samples)

def get_network(scope):
    preprocess_conf(conf)
    network = Network(scope, conf, 42, 42, 1)
    return network
