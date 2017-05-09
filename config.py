class AgentConfig(object):
  scale = 10000
  display = False

  max_step = 5000 * scale
  memory_size = 100 * scale

  batch_size = 32
  random_start = 1
  cnn_format = 'NCHW'
  discount = 0.99
  target_q_update_step = 1 * scale
  learning_rate = 0.00025
  learning_rate_minimum = 0.00025
  learning_rate_decay = 0.96
  learning_rate_decay_step = 5 * scale

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size

  history_length = 4
  train_frequency = 4
  learn_start = 5. * scale

  min_delta = -1
  max_delta = 1

  double_q = False
  dueling = False

  _test_step = 5 * scale
  _save_step = _test_step * 10

  vae_learning_rate = 0.0001

  beta = 0.3
  psc_start = int(2.5 * scale)
  bonus_preserve = 5
  # ae_start = 5 * scale
  max_ep_steps = 10000
  # ae_learn_start = scale
  # ae_avg_loss_interval = scale
  # ae_memory_size = ae_avg_loss_interval
  # ae_threshold = 0.01
  # play_interval = 10 * scale
  # latent_size = 256
  # img_scale = 255.


class EnvironmentConfig(object):
  env_name = 'MontezumaRevenge-v0'

  screen_width  = 84
  screen_height = 84

  # 42 by 42.
  ae_screen_height = 42
  ae_screen_width = 42

  max_reward = 1.
  min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'simple'
  action_repeat = 1

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  elif FLAGS.model == 'm2':
    config = M2

  for k, v in FLAGS.__dict__['__flags'].items():
    if k == 'gpu':
      if v == False:
        config.cnn_format = 'NHWC'
      else:
        config.cnn_format = 'NCHW'

    if hasattr(config, k):
      setattr(config, k, v)

  return config
