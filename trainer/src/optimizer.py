import numpy as np

rng = np.random.default_rng()

class Optimizer:

    def __init__(self, params, lamb):
        self.params = params
        self.n = len(params)
        # self.mu = lamb / 7
        self.mu = 2
        self.lamb = lamb
        self.sigma = 0.001

        self.noise_table = []
        self.w = np.array([np.log(self.mu + 0.5) - np.log(i) for i in range(1, self.mu + 1)])
        self.w /= np.sum(self.w)

    def getParams(self):
        noise = rng.normal(size=self.n)
        self.noise_table.append(noise)
        params = self.params + self.sigma * noise
        return params, len(self.noise_table) - 1

    def update(self, rewards):
        sorting = np.array(rewards).argsort()[::-1][:self.mu]
        step = np.zeros(self.n)
        for i in range(self.mu):
            step += self.w[i] * self.noise_table[sorting[i]]
        step *= self.sigma
        self.params += step
        self.noise_table.clear()