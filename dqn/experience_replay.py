"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""

import numpy as np
import time

floatX = 'float32'

class DataSet(object):
    """A replay memory consisting of circular buffers for observed images,
actions, and rewards.

    """
    def __init__(self, config, rng, data_format="NHWC"):
        """Construct a DataSet.

        Arguments:
            width, height - image size
            max_steps - the number of time steps to store
            phi_length - number of images to concatenate into a state
            rng - initialized numpy random number generator, used to
            choose random minibatches

        """
        # TODO: Specify capacity in number of state transitions, not

        self.width = config.screen_width
        self.height = config.screen_height
        self.max_steps = config.memory_size
        self.discount = config.discount
        self.phi_length = config.history_length
        self.rng = rng
        self.data_format = data_format

        # Allocate the circular buffers and indices.
        self.imgs = np.zeros((self.max_steps, self.height, self.width), dtype='float32')
        self.actions = np.zeros(self.max_steps, dtype='int32')
        self.rewards = np.zeros(self.max_steps, dtype=floatX)
        self.terminal = np.zeros(self.max_steps, dtype='bool')
        self.R = np.zeros(self.max_steps, dtype=floatX)

        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, img, action, reward, terminal):
        """Add a time step record.

        Arguments:
            img -- observed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.imgs[self.top] = img
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal
        self.R[self.top] = -1000.0

        if terminal:
            self.R[self.top] = reward
            idx = self.top
            count = 0
            while True:
                count += 1
                idx -= 1
                if self.terminal[idx]:
                    break
                self.R[idx] = self.R[idx+1]*self.discount + self.rewards[idx]

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps


    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - self.phi_length)

    def last_phi(self):
        """Return the most recent phi (sequence of image frames)."""
        indexes = np.arange(self.top - self.phi_length, self.top)
        phi = np.transpose(self.imgs.take(indexes, axis=0, mode='wrap'), [1, 2, 0])

        return phi

    def phi(self, img):
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img.

        """
        indexes = np.arange(self.top - self.phi_length + 1, self.top)

        phi = np.empty((self.phi_length, self.height, self.width), dtype=floatX)
        phi[0:self.phi_length - 1] = self.imgs.take(indexes,
                                                    axis=0,
                                                    mode='wrap')
        phi[-1] = img
        return phi

    def random_batch(self, batch_size):
        """Return corresponding imgs, actions, rewards, and terminal status for
batch_size randomly chosen state transitions.

        """
        # Allocate the response.

        imgs = np.zeros((batch_size,
                         self.height,
                         self.width,
                         self.phi_length + 1),
                        dtype='float32')

        actions = np.zeros(batch_size, dtype='int32')
        rewards = np.zeros(batch_size, dtype=floatX)
        terminal = np.zeros(batch_size, dtype='bool')
        # R is the Monte Carlo Return. :)
        R = np.zeros(batch_size, dtype=floatX)

        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            # index = self.rng.randint(self.bottom,
            #                          self.bottom + self.size - self.phi_length)
            index = self.rng.randint(0, self.size - self.phi_length)

            # Both the before and after states contain phi_length
            # frames, overlapping except for the first and last.
            all_indices = np.arange(index, index + self.phi_length + 1)
            end_index = index + self.phi_length - 1

            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but its last frame (the
            # second-to-last frame in imgs) may be terminal. If the last
            # frame of the initial state is terminal, then the last
            # frame of the transitioned state will actually be the first
            # frame of a new episode, which the Q learner recognizes and
            # handles correctly during training by zeroing the
            # discounted future reward estimate.
            if np.any(self.terminal.take(all_indices[0:-2], mode='wrap')) or self.R.take(end_index,
                                                                                         mode='wrap') == -1000.0:
                continue

            # Add the state transition to the response.
            imgs[count] = np.transpose(self.imgs.take(all_indices, axis=0, mode='wrap'), [1, 2, 0])
            actions[count] = self.actions.take(end_index, mode='wrap')
            rewards[count] = self.rewards.take(end_index, mode='wrap')
            terminal[count] = self.terminal.take(end_index, mode='wrap')
            R[count] = self.R.take(end_index, mode='wrap')
            count += 1
        if self.data_format == "NHWC":
          s_t = imgs[..., :self.phi_length]
          s_t_plus_1 = imgs[..., -self.phi_length:]
        else:
          imgs = np.transpose(imgs, [0, 3, 1, 2])
          s_t = imgs[:, :self.phi_length, ...]
          s_t_plus_1 = imgs[:, -self.phi_length:, ...]
        return s_t, s_t_plus_1, actions, rewards, terminal, R
