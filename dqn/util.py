import tensorflow as tf
import numpy as np
# import skimage.transform
from scipy.misc import imsave
# import cv2
# # kernel_height is the same as kernel_width by default
# # all default float32 DT_FLOAT
#
#
# def rgb2y(image, output_shape):
#     assert (len(image.shape) == 3)
#     assert (image.shape[-1] == 3)
#     grayscale_img = np.dot(image[..., :3], [0.229, 0.587, 0.144])
#     # grayscale_img = skimage.color.rgb2gray(image)
#     im_y = cv2.resize(grayscale_img, (84, 84))
#     im_y = cv2.resize(im_y, output_shape)
#     return im_y.astype(np.uint8)




def clipped_error(x, delta=1.):
    try:
        return tf.select(tf.abs(x) < delta, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < delta, 0.5 * tf.square(x), tf.abs(x) - 0.5)



def linear(x,
           output_size,
           name,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           weight_return=False):
    shape = x.get_shape().as_list() # need to be 1D vector except the batchsize dim.

    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], output_size], initializer = initializer)
        # w = tf.get_variable('w', [shape[1], output_size], tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(0.))

        out = tf.nn.bias_add(tf.matmul(x, w), b)

    if activation_fn is not None:
      out = activation_fn(out)

    if weight_return:
        return out,w, b
    return out


def conv2d(x,
           kernel_size,
           stride,
           output_channel,
           name,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding="VALID",
           weight_return=False):
    with tf.variable_scope(name):
        if data_format == 'NCHW':
            _stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_channel]
        elif data_format == 'NHWC':
            _stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_channel]

        w = tf.get_variable("w", kernel_shape, initializer=initializer)
        conv = tf.nn.conv2d(x, w, strides=_stride, padding=padding, data_format=data_format)

        b = tf.get_variable("b", [output_channel], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)

    if activation_fn is not None:
        out = activation_fn(out)

    if weight_return:
        return out, w, b

    return out


def deconv2d(x,
             kernel_size,
             stride,
             output_shape, # not kernel this time. Maybe would pad the size if not big enough or just clip.
             name,
             initializer=tf.truncated_normal_initializer(stddev=0.02),
             activation_fn=tf.nn.relu,
             data_format='NHWC',
             padding='VALID', # Can valid help?
             weight_return=False):
    # output shape is fixed previously

    with tf.variable_scope(name):
        # h, w, out, in
        # NHWC
        # kernel's size is the same as the forward's process. (compute like the gradient coputation)
        # but the bias is not matched
        if data_format == 'NCHW':
            _stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], output_shape[1], x.get_shape()[1]]
        elif data_format == 'NHWC':
            _stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], output_shape[-1], x.get_shape()[-1]]

        w = tf.get_variable("w", kernel_shape, initializer=initializer)
        convt = tf.nn.conv2d_transpose(x, w, output_shape = output_shape, strides = _stride,
                                       padding = padding, data_format = data_format)

        b = tf.get_variable("b", [kernel_shape[-2]], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(convt, b, data_format)

    if activation_fn is not None:
        out = activation_fn(out)

    if weight_return:
        return out, w, b

    return out