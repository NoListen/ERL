from logging import getLogger

import tensorflow as tf

from ops import *
from utils import *

logger = getLogger(__name__)

class Network:

  def __init__(self, conf, height, width, num_channels, dens_scope, device = "/cpu:0"):
    logger.info("Building gated_pixel_cnn starts")
    self._device = device

    self.height, self.width, self.channel = height, width, num_channels
    self.pixel_depth = 256
    self.q_levels = q_levels = conf.q_levels

    self.inputs = tf.placeholder(tf.float32, [None, height, width, num_channels]) # [N,H,W,C]
    self.target_pixels = tf.placeholder(tf.int64, [None, height, width, num_channels]) # [N,H,W,C] (the index of a one-hot representation of D)
    self.dens_scope = dens_scope

    self.index_range = np.arange(height * width)

    self.learning_rate = conf.learning_rate
    self.momentum = conf.momentum
    self.decay = conf.decay
    self.epsilon = conf.epsilon


    # input conv layer
    logger.info("Building CONV_IN")    
    net = conv(self.inputs, conf.gated_conv_num_feature_maps, [7, 7], "A", num_channels, scope="CONV_IN")
    
    # main gated layers
    for idx in xrange(conf.gated_conv_num_layers):
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
    self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    #
    # for i in self.vars:
    # 	print i.name
    #
    grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list = self.vars)
    self.new_grads_and_vars = \
        [(tf.clip_by_value(gv[0], -conf.grad_clip, conf.grad_clip), gv[1]) for gv in grads_and_vars]

    self.train_op = self.optimizer.apply_gradients(self.new_grads_and_vars)
    # grads = tf.gradients(self.loss, self.vars)
    # self.grads, _ = tf.clip_by_global_norm(grads, conf.grad_clip)

    # grads_and_vars = list(zip(self.grads, self.vars))
    # self.train_op = self.optimizer.apply_gradients(grads_and_vars)

    show_all_variables()

    logger.info("Building gated_pixel_cnn finished")

  # def apply_gradients(self, target_network):
  #   self.apply_optim = tf.train.RMSPropOptimizer(self.learning_rate, self.decay, self.momentum, self.epsilon)
  #   grads_and_vars = list(zip(self.grads, target_network.get_vars()))
  #   apply_gradients_op = self.apply_optim.apply_gradients(grads_and_vars)
  #   return apply_gradients_op

  def get_vars(self):
    return self.vars

  def predict(self, sess, images):
    '''
    images # shape [N,H,W,C]
    returns predicted image # shape [N,H,W,C]
    '''
    # self.output shape [NHWC,D]
    pixel_value_probabilities = sess.run(self.output, {self.inputs: images}) # shape [N,H,W,C,D], values [probability distribution]
    
    # argmax or random draw # [NHWC,1]  quantized index - convert back to pixel value    
    pixel_value_indices = np.argmax(pixel_value_probabilities, 4) # shape [N,H,W,C], values [index of most likely pixel value]
    pixel_values = np.multiply(pixel_value_indices, ((self.pixel_depth - 1) / (self.q_levels - 1))) #shape [N,H,W,C]

    return pixel_values

  # def test(self, sess, mages, with_update=False):
  #   if with_update:
  #     _, cost = sess.run([self.optim, self.loss], 
  #                             { self.inputs: images[0], self.target_pixels: images[1] })
  #   else:
  #     cost = sess.run(self.loss, { self.inputs: images[0], self.target_pixels: images[1] })
  #   return cost

  def generate_from_occluded(self, images, num_generated_images, occlude_start_row):
    samples = np.copy(images[0:num_generated_images,:,:,:])
    # samples[:,occlude_start_row:,:,:] = 0.

    # for i in xrange(occlude_start_row,self.height):
    #   for j in xrange(self.width):
    #     for k in xrange(self.channel):
    next_sample = self.predict(samples) / (self.pixel_depth - 1.) # argmax or random draw here
    # samples[:, i, j, k] = next_sample

    return next_sample

  def prob_evaluate(self, sess, images, with_update=False):
    if with_update:
      _, indexes, target = sess.run([self.train_op, self.target_pixels_loss, self.flattened_output],
                                   feed_dict = {
                                     self.inputs: images[0],
                                     self.target_pixels: images[1]
                                   })
    else:
      indexes, target  = sess.run([self.target_pixels_loss, self.flattened_output],
                                  feed_dict = {
                                    self.inputs: images[0],
                                     self.target_pixels: images[1]
                                  })

    pred_prob = target[self.index_range, indexes]

    # sample = self.generate_samples(sess,  images[0])
    # # print t
    # if t > 5000 and not with_update:
    #   save_images(np.concatenate((images[0], sample), axis=2),
    #                     42, 42 * 2, 1, 1,
    #                     directory="pixelcnn/" , prefix="global_t%i" % t)
    # p = np.prod(pred_prob)
    return pred_prob


  def generate_samples(self, sess, image):
    # samples[:,occlude_start_row:,:,:] = 0.

    # for i in xrange(occlude_start_row,self.height):
    #   for j in xrange(self.width):
    #     for k in xrange(self.channel):
    next_sample = self.predict(sess, image) / (self.pixel_depth - 1.) # argmax or random draw here
    # samples[:, i, j, k] = next_sample

    return next_sample


  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.op_scope([], name, "GatedPixelCNN") as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)