import numpy as np
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf

rng = np.random.default_rng()

class Optimizer:

    def __init__(self, params, lamb):
        self.params = params
        self.n = len(params)
        # self.mu = lamb / 7
        self.mu = 1
        self.lamb = lamb
        self.sigma = 0.01

        self.noise_table = []
        self.w = np.array([np.log(self.mu + 0.5) - np.log(i) for i in range(1, self.mu + 1)])
        self.w /= np.sum(self.w)

        self.generation = 0

        logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_writer = tf.summary.create_file_writer(logdir + "/evolution")
        self.log_writer.set_as_default()

    def getParams(self):
        noise = rng.normal(size=self.n)
        self.noise_table.append(noise)
        params = self.params + self.sigma * noise
        return params #, len(self.noise_table) - 1

    def update(self, rewards):
        sorting = np.array(rewards).argsort()[::-1][:self.mu]
        step = np.zeros(self.n)
        for i in range(self.mu):
            step += self.w[i] * self.noise_table[sorting[i]]
        step *= self.sigma
        self.params += step
        self.noise_table.clear()

        print("Generation ", self.generation)
        tf.summary.scalar('top reward', data=rewards[sorting[0]], step=self.generation)

        self.generation += 1