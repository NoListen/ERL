import random
import tensorflow as tf
import numpy as np
# from dqn.agent import Agent
from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config
from dqn.AutoEncoder.ae import AutoEncoder
import argparse
from dqn.utils import imresize
import os
from dqn.utils import loadFromFlat
from scipy.misc import imsave
from tqdm import tqdm

def set_random_seed(seed):
  tf.set_random_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def concat2imgs(img1, img2):
    img = np.zeros((84, 42))
    img[:42, :] = img1
    img[42:, :] = img2
    return img

def saveimg(img, step, prefix):
    imsave(prefix+"_%i.png" % step, img)

class RandomPolicy(object):
    def __init__(self, na):
        self.n = na

    def action(self, _):
        return np.random.randint(0, self.n)


def main(_):
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--seed', help='RNG seed', type=int, default=123)
  parser.add_argument("--use-gpu", action="store_true")
  parser.add_argument("--mode", help="Bonus mode", default="autoencoder")
  parser.add_argument("--model-dir", help="the path of the model", default="ae_model/model.p")
  parser.add_argument("--img-dir", help="the path to save image", default="imgs/")
  parser.add_argument("--n", help="the number of episodes", default=10)

  args = parser.parse_args()

  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  config.gpu_options.allow_growth=True

  with tf.Session(config=config) as sess:
    config = get_config(args)

    if config.env_type == 'simple':
      env = SimpleGymEnvironment(config)
    else:
      env = GymEnvironment(config)

    if not tf.test.is_gpu_available() and args.use_gpu:
      raise Exception("use_gpu flag is true when no GPUs are available")

    # Build the density model
    density_model = AutoEncoder("ae", sess, config)
    loadFromFlat(density_model.get_variables(), args.model_dir)

    na = config.n_action
    last_screen, reward, action, terminal = env.new_random_game()
    last_screen42x42 = imresize(last_screen, (42, 42), order=1)
    pi = RandomPolicy(na)

    if not os.path.exists(args.img_dir):
        os.mkdir(args.img_dir)

    # At first, use random action taker.
    for i in tqdm(range(args.n)):
        ep_steps = 0
        prefix = args.img_dir + "ep%i/" % i
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        prefix = prefix + 'img'
        while True:
            action = pi.action(last_screen)
            screen, reward, terminal = env.act(action, is_training=True)
            screen42x42 = imresize(screen, (42, 42), order=1)

            oh_action = np.zeros(na)
            oh_action[action] = 1

            density_model.memory.add_sample(last_screen42x42, action, terminal)
            ep_steps += 1

            if ep_steps >= 4:
                pscreen42x42 = density_model.predict().reshape(42, 42)
                img = concat2imgs(screen42x42, pscreen42x42)
                saveimg(img, ep_steps, prefix)

            # Update
            last_screen42x42 = screen42x42
            last_screen = screen

            if terminal:
                last_screen, reward, action, terminal = env.new_random_game()
                last_screen42x42 = imresize(last_screen, (42, 42), order=1)
                break




if __name__ == '__main__':
  tf.app.run()
