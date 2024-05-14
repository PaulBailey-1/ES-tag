import numpy as np
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf

rng = np.random.default_rng()

def normalize(vec):
    l = np.linalg.norm(vec)
    if l == 0:
        return vec
    return [x / l for x in vec]
class Optimizer:

    def __init__(self, params, lamb, logDir, config=None):
        self.params = params
        self.n = len(params)
        # self.mu = lamb / 7
        self.lamb = lamb
        self.mu = 1
        self.sigma = 0.01

        self.step = np.zeros(self.n)
        self.lastStep = np.zeros(self.n)

        if config:
            self.mu = config['mu']
            self.sigma = config['sigma']
            if self.mu > self.lamb:
                print("Error: Parent population cannot be larger than total population")

        self.noise_table = []
        self.w = np.array([np.log(self.mu + 0.5) - np.log(i) for i in range(1, self.mu + 1)])
        self.w /= np.sum(self.w)

        self.log_writer = tf.summary.create_file_writer(logDir + "/evolution")
        self.log_writer.set_as_default()

        print(f"Initialized optimizer n: {self.n} mu: {self.mu} sigma: {self.sigma}")

    def getParams(self):
        noise = rng.normal(size=self.n)
        self.noise_table.append(noise)
        params = self.params + self.sigma * noise
        return params #, len(self.noise_table) - 1

    def update(self, generation, rewards):
        sorting = np.array(rewards).argsort()[::-1][:self.mu]
        for i in range(self.mu):
            self.step += self.w[i] * self.noise_table[sorting[i]]
        self.step *= self.sigma
        self.params += self.step
        self.noise_table.clear()

        avgReward = 0
        for i in sorting:
            avgReward += rewards[i]
        avgReward /= len(sorting)

        stepCorrelation = np.dot(normalize(self.step), normalize(self.lastStep))
        self.lastStep = np.copy(self.step)
        print(stepCorrelation)
            
        tf.summary.scalar('top reward', data=rewards[sorting[0]], step=generation)
        tf.summary.scalar('mean reward', data=avgReward, step=generation)
        tf.summary.scalar('step size', data=self.sigma, step=generation)
        tf.summary.scalar('step correlation', data=stepCorrelation, step=generation)
