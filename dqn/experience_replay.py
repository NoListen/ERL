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
        # number of saved time steps.

        # Store arguments.
        self.width = config.screen_width
        self.height = config.screen_height
        self.max_steps = config.memory_size
        self.discount = config.discount
        self.phi_length = config.history_length
        self.rng = rng
        self.data_format = data_format

        # Allocate the circular buffers and indices.
        self.imgs = np.zeros((self.max_steps, self.height, self.width), dtype='uint8')
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
                # print self.rewards[idx]
            # print "new R computed", count, "steps", self.R[idx+1]
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

        # share the same part. This state and the next state.
        imgs = np.zeros((batch_size,
                         self.height,
                         self.width,
                         self.phi_length + 1),
                        dtype='uint8')

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

        return imgs, actions, rewards, terminal, R

# TESTING CODE BELOW THIS POINT...

def simple_tests():
    np.random.seed(222)
    dataset = DataSet(width=2, height=3,
                      rng=np.random.RandomState(42),
                      max_steps=6, phi_length=4)
    for i in range(10):
        img = np.random.randint(0, 256, size=(3, 2))
        action = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True
        print 'img', img
        dataset.add_sample(img, action, reward, terminal)
        print "I", dataset.imgs
        print "A", dataset.actions
        print "R", dataset.rewards
        print "T", dataset.terminal
        print "SIZE", dataset.size
        print
    print "LAST PHI", dataset.last_phi()
    print
    print 'BATCH', dataset.random_batch(2)


def speed_tests():

    dataset = DataSet(width=80, height=80,
                      rng=np.random.RandomState(42),
                      max_steps=20000, phi_length=4)

    img = np.random.randint(0, 256, size=(80, 80))
    action = np.random.randint(16)
    reward = np.random.random()
    start = time.time()
    for i in range(100000):
        terminal = False
        if np.random.random() < .05:
            terminal = True
        dataset.add_sample(img, action, reward, terminal)
    print "samples per second: ", 100000 / (time.time() - start)

    start = time.time()
    for i in range(200):
        dataset.random_batch(32)
    print "batches per second: ", 200 / (time.time() - start)

    print dataset.last_phi()


def trivial_tests():

    dataset = DataSet(width=2, height=1,
                      rng=np.random.RandomState(42),
                      max_steps=3, phi_length=2)

    img1 = np.array([[1, 1]], dtype='uint8')
    img2 = np.array([[2, 2]], dtype='uint8')
    img3 = np.array([[3, 3]], dtype='uint8')

    dataset.add_sample(img1, 1, 1, False)
    dataset.add_sample(img2, 2, 2, False)
    dataset.add_sample(img3, 2, 2, True)
    print "last", dataset.last_phi()
    print "random", dataset.random_batch(1)


def max_size_tests():
    dataset1 = DataSet(width=3, height=4,
                      rng=np.random.RandomState(42),
                      max_steps=10, phi_length=4)
    dataset2 = DataSet(width=3, height=4,
                      rng=np.random.RandomState(42),
                      max_steps=1000, phi_length=4)
    for i in range(100):
        img = np.random.randint(0, 256, size=(4, 3))
        action = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True
        dataset1.add_sample(img, action, reward, terminal)
        dataset2.add_sample(img, action, reward, terminal)
        np.testing.assert_array_almost_equal(dataset1.last_phi(),
                                             dataset2.last_phi())
        print "passed"


def test_memory_usage_ok():
    import memory_profiler
    dataset = DataSet(width=80, height=80,
                      rng=np.random.RandomState(42),
                      max_steps=100000, phi_length=4)
    last = time.time()

    for i in xrange(1000000000):
        if (i % 100000) == 0:
            print i
        dataset.add_sample(np.random.random((80, 80)), 1, 1, False)
        if i > 200000:
            imgs, actions, rewards, terminals = \
                                        dataset.random_batch(32)
        if (i % 10007) == 0:
            print time.time() - last
            mem_usage = memory_profiler.memory_usage(-1)
            print len(dataset), mem_usage
        last = time.time()


def main():
    speed_tests()
    test_memory_usage_ok()
    max_size_tests()
    simple_tests()

if __name__ == "__main__":
    main()