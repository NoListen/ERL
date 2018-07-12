import time
import _pickle as cPickle
import numpy as np
import tensorflow as tf
from skimage.transform import resize
import pickle

imresize = resize

# the input is an numpy array
def onehot_actions(actions):
    n = len(actions)
    oh_actions = np.zeros((n, 18))
    oh_actions[np.arange(n), actions] = 1
    return oh_actions

def rgb2gray(image):
  return np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype('uint8')

def timeit(f):
  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()

    print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
    return result
  return timed

def get_time():
  return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

@timeit
def save_pkl(obj, path):
  with open(path, 'w') as f:
    cPickle.dump(obj, f)
    print("  [*] save %s" % path)

@timeit
def load_pkl(path):
  with open(path) as f:
    obj = cPickle.load(f)
    print("  [*] load %s" % path)
    return obj

@timeit
def save_npy(obj, path):
  np.save(path, obj)
  print("  [*] save %s" % path)

@timeit
def load_npy(path):
  obj = np.load(path)
  print("  [*] load %s" % path)
  return obj

def saveToFlat(var_list, param_pkl_path):
    # get all the values
    var_values = np.concatenate([v.flatten() for v in tf.get_default_session().run(var_list)])
    pickle.dump(var_values, open(param_pkl_path, "wb"))

def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params.astype(np.float32)

def loadFromFlat(var_list, param_pkl_path):
    flat_params = load_from_file(param_pkl_path)
    print("the type of the parameters stored is ", flat_params.dtype)
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        print(v.name)
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})