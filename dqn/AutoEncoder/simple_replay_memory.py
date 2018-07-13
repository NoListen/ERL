"""
This if for autoencoder not the RL-agent to learn
Just Serve as one buffer
"""
import os
import random
import logging
import numpy as np

from dqn.utils import save_npy, load_npy


# At first fill the replay buffer, then sample to learn.
import numpy as np

class SimpleDataSet(object):
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

        self.width = config.ae_screen_width
        self.height = config.ae_screen_height
        self.max_steps = config.ae_memory_size

        self.phi_length = config.history_length
        self.rng = rng
        self.data_format = data_format

        # the memory to store is in float format.
        self.imgs = np.zeros((self.max_steps, self.height, self.width), dtype='float32')
        self.actions = np.zeros(self.max_steps, dtype='int32')
        self.terminal = np.zeros(self.max_steps, dtype='bool')

        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, img, action, terminal):
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
        self.terminal[self.top] = terminal

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

    def last_action(self):
        index = (self.top - 1 + self.size) % self.size
        return self.actions[index]


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


            if np.any(self.terminal.take(all_indices[:-1], mode='wrap')):
                continue

            # Add the state transition to the response.
            imgs[count] = np.transpose(self.imgs.take(all_indices, axis=0, mode='wrap'), [1, 2, 0])
            actions[count] = self.actions.take(end_index, mode='wrap')
            count += 1
        if self.data_format == "NHWC":
          s_t = imgs[..., :self.phi_length]
          s_t_plus_1 = imgs[..., -1]
        else:
          imgs = np.transpose(imgs, [0, 3, 1, 2])
          s_t = imgs[:, :self.phi_length, ...]
          s_t_plus_1 = imgs[:, -1, ...]
        return s_t, s_t_plus_1, actions
