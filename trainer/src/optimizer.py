from src.logger import log
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
        self.x = params
        self.n = len(params)
        # self.mu = lamb / 7
        self.lamb = lamb
        self.mu = 1
        self.delta = 0.01
        self.c = 1 / np.sqrt(self.n)
        self.ccov = 2 / self.n ** 2
        self.beta = 1 / self.n
        self.chiHat = np.sqrt(self.n) * (1 - (1 / (4 * self.n)) + (1 / (21 * self.n ** 2)))

        self.z = np.zeros(self.n)
        self.B = np.identity(self.n)
        self.C = np.identity(self.n)
        self.s = np.zeros(self.n)
        self.sNorm = np.zeros(self.n)
        self.step = np.zeros(self.n)
        self.lastStep = np.zeros(self.n)
        self.parents = np.zeros((self.mu, self.n))
        self.parents = np.zeros((self.mu, self.n))

        self.firstGen = True

        if config:
            if 'mu' in config: self.mu = config['mu']
            if 'delta0' in config: self.delta = config['delta0']
            if 'c' in config: self.c = config['c']
            if 'ccov' in config: self.ccov = config['ccov']
            if 'beta' in config: self.beta = config['beta']
            if 'chihat' in config: self.chihat = config['chihat']

            if self.mu > self.lamb:
                log("Error: Parent population cannot be larger than total population")

        self.noise_table = []
        self.w = np.array([np.log(self.mu + 0.5) - np.log(i) for i in range(1, self.mu + 1)])
        self.w /= np.sum(self.w)

        self.log_writer = tf.summary.create_file_writer(logDir + "/evolution")
        self.log_writer.set_as_default()

        log(f"Initialized optimizer n: {self.n} lambda: {lamb} mu: {self.mu} delta: {self.delta}")

    def getParams(self):
        # if not self.firstGen and len(self.noise_table) < self.mu:
        #     noise = self.parentNoises[len(self.noise_table)]
        # else:
        noise = rng.normal(size=self.n)
        self.noise_table.append(noise)
        params = self.x + self.delta * np.dot(self.B, noise)
        return params
    

    # CMA-ES: https://ieeexplore.ieee.org/document/542381
    def update(self, generation, rewards):

        sorting = np.array(rewards).argsort()[::-1][:self.mu]
        # self.parents = [self.x + self.delta * np.dot(self.B, self.noise_table[i]) for i in sorting]

        # Compute step, add to mean
        self.z = np.zeros(self.n)
        for i in range(self.mu):
            self.z += self.w[i] * self.noise_table[sorting[i]]
        self.step = self.delta * np.dot(self.B, self.z)
        self.x += self.step
        self.noise_table.clear()

        # CMA && CSA
        eigvals, eigvecs = np.linalg.eigh(self.C)
        Bnorm = np.copy(eigvecs)
        for i in range(self.n):
            eigvecs[:,i] *= np.sqrt(eigvals[i])
        self.B = eigvecs
        cu = np.sqrt(self.c * (2 - self.c))

        self.s = (1 - self.c) * self.s + cu * np.dot(self.B, self.z)
        self.sNorm = (1 - self.c) * self.sNorm + cu * np.dot(Bnorm, self.z)

        self.C = (1 - self.ccov) * self.C + self.ccov * np.dot(self.s, self.s.T)
        self.delta = self.delta * np.exp(self.beta * (np.linalg.norm(self.sNorm) - self.chiHat))

        # Calculate parent noise equivelents for u + l
        # Binv = np.linalg.inv(self.B)
        # self.parentNoises = [np.dot(Binv, (y - self.x) / self.delta) for y in self.parents]

        # Logging
        avgReward = 0
        for x in rewards:
            avgReward += x
        avgReward /= len(rewards)
        # for i in sorting:
        #     avgReward += rewards[i]
        # avgReward /= len(sorting)

        stepCorrelation = np.dot(normalize(self.step), normalize(self.lastStep))
        self.lastStep = np.copy(self.step)
            
        tf.summary.scalar('top reward', data=rewards[sorting[0]], step=generation)
        tf.summary.scalar('mean reward', data=avgReward, step=generation)
        tf.summary.scalar('delta', data=self.delta, step=generation)
        tf.summary.scalar('path length', data=np.linalg.norm(self.sNorm), step=generation)
        tf.summary.scalar('step correlation', data=stepCorrelation, step=generation)

        self.firstGen = False
