class AgentConfig(object):
  scale = 2000
  display = False

  max_step = 5000 * scale
  memory_size = 100 * scale

  batch_size = 32
  random_start = 1
  #cnn_format = 'NCHW'
  cnn_format = 'NHWC'
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

  double_q = True
  dueling = False

  _test_step = 2 * scale
  _save_step = _test_step * 10

  max_ep_steps = 10000

  backend = 'tf'
  env_type = 'simple'
  action_repeat = 2


class EnvironmentConfig(object):
  env_name = 'MontezumaRevenge-v0'

  screen_width  = 84
  screen_height = 84


  max_reward = 1.
  min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
  pass

class PixelCNNConfig(AgentConfig, EnvironmentConfig):
  beta = 0.1
  psc_scale = 0.1

  psc_start = int(2.5 * AgentConfig.scale)
  psc_sample_ratio = 0.25

class AEConfig(AgentConfig, EnvironmentConfig):
  beta = 0.1

  bonus_scale = 0.1
  ae_start = 5 * AgentConfig.scale
  ae_learn_start = AgentConfig.scale
  ae_avg_loss_interval = AgentConfig.scale
  ae_memory_size = ae_avg_loss_interval
  ae_threshold = 0.01
  latent_size = 256
  n_action = 18 # ha
  # rd coded

  ae_screen_width = 42
  ae_screen_height = 42

  ae_batch_size = 64
  ae_train_steps = 32

  ae_learning_rate = 0.0001

  ae_model_path = "ae_model"
  ae_save_step = 5*Agent.Config + 1

def get_config(args):
  if args.mode == "pixelcnn":
    config = PixelCNNConfig
  elif args.mode == "autoencoder":
    config = AEConfig
  else:
    config = DQNConfig

  # if args.use_gpu:
  #   config.cnn_format = 'NHWC'
  # else:
  #   config.cnn_format = 'NCHW'

  return config
