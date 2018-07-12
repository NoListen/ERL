import random
import tensorflow as tf
import numpy as np
# from dqn.agent import Agent
from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config
import argparse

def set_random_seed(seed):
  tf.set_random_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

def main(_):
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--seed', help='RNG seed', type=int, default=123)
  parser.add_argument('--test', action="store_true")
  parser.add_argument("--use-gpu", action="store_true")
  parser.add_argument("--mode", help="Bonus mode", default="pixelcnn")
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

    if args.mode == "pixelcnn":
      from dqn.agent import Agent
      agent = Agent(config, env, sess)
    else:
      from dqn.agent_model import Agent
      agent = Agent(config, env, sess)

    print("CNN format", config.cnn_format)
    if not args.test:
      print("training ...")
      agent.train()
    else:
      print("testing ...")
      agent.play()

if __name__ == '__main__':
  tf.app.run()
