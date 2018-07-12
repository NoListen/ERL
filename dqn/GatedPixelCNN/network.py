from logging import getLogger

import tensorflow as tf

from .ops import *
from .utils import *

logger = getLogger(__name__)

class Network:
  def __init__(self, scope, *args, **kargs):
    with tf.variable_scope(scope):
      self._init(*args, **kargs)
      self.scope = scope

  def _init(self, conf, height, width, num_channels, device = "/cpu:0"):
    logger.info("Building gated_pixel_cnn starts")
    self._device = device

    self.height, self.width, self.channel = height, width, num_channels
    self.pixel_depth = 256
    self.q_levels = q_levels = conf.q_levels

    self.inputs = tf.placeholder(tf.float32, [None, height, width, num_channels]) # [N,H,W,C]
    self.target_pixels = tf.placeholder(tf.int64, [None, height, width, num_channels]) # [N,H,W,C] (the index of a one-hot representation of D)

    self.index_range = np.arange(height * width)

    self.learning_rate = conf.learning_rate
    self.momentum = conf.momentum
    self.decay = conf.decay
    self.epsilon = conf.epsilon


    # input conv layer
    logger.info("Building CONV_IN")    
    net = conv(self.inputs, conf.gated_conv_num_feature_maps, [7, 7], "A", num_channels, scope="CONV_IN")
    
    # main gated layers
    for idx in range(conf.gated_conv_num_layers):
      scope = 'GATED_CONV%d' % idx
      net = gated_conv(net, [1, 1], num_channels, scope=scope)
      logger.info("Building %s" % scope)

    # output conv layers
    net = tf.nn.relu(conv(net, conf.output_conv_num_feature_maps, [1, 1], "B", num_channels, scope='CONV_OUT0'))
    self.logits = tf.nn.relu(conv(net, q_levels * num_channels, [1, 1], "B", num_channels, scope='CONV_OUT1')) # shape [N,H,W,DC]

    if (num_channels > 1):
      self.logits = tf.reshape(self.logits, [-1, height, width, q_levels, num_channels]) # shape [N,H,W,DC] -> [N,H,W,D,C]            
      self.logits = tf.transpose(self.logits, perm=[0, 1, 2, 4, 3]) # shape [N,H,W,D,C] -> [N,H,W,C,D]             
    
    flattened_logits = tf.reshape(self.logits, [-1, q_levels]) # [N,H,W,C,D] -> [NHWC,D] 
    self.target_pixels_loss = tf.reshape(self.target_pixels, [-1]) # [N,H,W,C] -> [NHWC]
    
    logger.info("Building loss and optims")    
    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
       labels=self.target_pixels_loss, logits = flattened_logits))


    # If you want to do the softmax, you have better to make the tensor 2D
    self.flattened_output = tf.nn.softmax(flattened_logits) #shape [NHWC,D], values [probability distribution].
    self.output = tf.reshape(self.flattened_output, [-1, height, width, num_channels, q_levels]) #shape [N,H,W,C,D], values [probability distribution]

    self.optimizer = tf.train.RMSPropOptimizer(conf.learning_rate, conf.decay, conf.momentum, conf.epsilon)
    self.vars = self.get_trainable_variables()

    grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list = self.vars)
    self.new_grads_and_vars = \
        [(tf.clip_by_value(gv[0], -conf.grad_clip, conf.grad_clip), gv[1]) for gv in grads_and_vars]

    self.train_op = self.optimizer.apply_gradients(self.new_grads_and_vars)

    # show_all_variables()
    logger.info("Building gated_pixel_cnn finished")

  def generate_from_occluded(self, images, num_generated_images, occlude_start_row):
    samples = np.copy(images[0:num_generated_images,:,:,:])

    next_sample = self.predict(samples) / (self.pixel_depth - 1.) # argmax or random draw here
    # samples[:, i, j, k] = next_sample

    return next_sample

  def prob_evaluate(self, sess, images, with_update=False):
    if with_update:
      _, indexes, target = sess.run([self.train_op, self.target_pixels_loss, self.flattened_output],
                                   feed_dict = {
                                     self.inputs: images,
                                     self.target_pixels: images
                                   })
    else:
      indexes, target  = sess.run([self.target_pixels_loss, self.flattened_output],
                                  feed_dict = {
                                    self.inputs: images,
                                     self.target_pixels: images
                                  })

    pred_prob = target[self.index_range, indexes]
    return pred_prob

  # Used in A3C.
  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_trainable_variables()
    dst_vars = self.get_trainable_variables()

    sync_ops = []
    with tf.device(self._device):
      with tf.op_scope([], name, "GatedPixelCNN") as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  def get_variables(self):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

  def get_trainable_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)