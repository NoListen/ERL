"""
This if for autoencoder not the RL-agent to learn
Just Serve as one buffer
"""
import os
import random
import logging
import numpy as np

from dqn.utils import save_npy, load_npy


class SimpleReplayMemory:
    def __init__(self, config):

        self.cnn_format = config.cnn_format
        self.memory_size = config.ae_memory_size
        self.screens = np.empty((self.memory_size, config.ae_screen_height, config.ae_screen_width), dtype=np.float16)
        self.dims = (config.ae_screen_height, config.ae_screen_width)
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0

        # pre-allocate prestates and poststates for minibatch
        self.states = np.empty((self.batch_size, 1) + self.dims, dtype=np.float16)

    def add(self, screen):
        assert screen.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.screens[self.current, ...] = screen
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index %= self.count
        # if is not in the beginning of matrix
        return self.screens[index, ...]

    def sample(self):
        # memory must include poststate, prestate and history
        assert self.count > self.batch_size
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(0, self.count - 1)
                # if wraps over current pointer, then get new one
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            # for i in range(self.batch_size):
            self.states[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        if self.cnn_format == 'NHWC':
            return self.states.reshape((-1,) + self.dims+(1,))
        return self.states

    def train_samples(self):
        index = 0
        while index < self.count:
            self.states = self.screens[index:min(index+self.batch_size, self.count), ...] # batch size at first.
            index += self.batch_size
            if self.cnn_format == "NHWC":
                self.states = self.states.reshape((-1,) + self.dims+(1,))
                yield self.states
            yield self.states


