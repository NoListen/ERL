from .ops import linear, conv2d, clipped_error
import tensorflow as tf



# Nature DQN.
class DQN(object):
    def __init__(self, name, *args, **kargs):
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            self._init(*args, **kargs)

    def _init(self, screen_shape,
              history_length, num_action,
              initializer, activation_fn,
              cnn_format, dueling):
        with tf.variable_scope('prediction'):
            if cnn_format == 'NHWC':
                self.s_t = tf.placeholder('float32', (None,) + screen_shape + (history_length,), name='s_t')
            else:
                self.s_t = tf.placeholder('float32', (None, history_length) + screen_shape, name='s_t')

            last_out = self.s_t
            last_out = conv2d(last_out, 32, [8, 8], [4, 4], initializer, activation_fn, cnn_format,
                                 name='l1', variable_return=False)
            last_out = conv2d(last_out, 64, [4, 4], [2, 2], initializer, activation_fn, cnn_format,
                                 name='l2', variable_return=False)
            last_out = conv2d(last_out, 64, [3, 3], [1, 1], initializer, activation_fn, cnn_format,
                                  name='l3', variable_return=False)

            shape = last_out.get_shape().as_list()
            last_out = tf.reshape(last_out, [-1, reduce(lambda x, y: x * y, shape[1:])])

            if dueling:
                value_hid = linear(last_out, 512, activation_fn=activation_fn, name='value_hid',
                                         variable_return=False)

                adv_hid = linear(last_out, 512, activation_fn=activation_fn, name='adv_hid',
                                       variable_return=False)

                # No activation for the two
                value = linear(value_hid, 1, name='value_out', variable_return=False)
                advantage = linear(adv_hid, num_action, name='adv_out', variable_return=False)

                self.q = value + (advantage -
                             tf.reduce_mean(advantage, reduction_indices=1, keep_dims=True))
            else:
                l4, _, _ = linear(last_out, 512, activation_fn=activation_fn, name='l4')
                self.q, _, _ = linear(l4, num_action, name='q')

            self.q_action = tf.argmax(self.q, dimension=1)

    def

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
