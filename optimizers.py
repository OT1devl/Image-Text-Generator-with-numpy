import numpy as np

class Optimizer:
    def __init__(self):
        raise NotImplementedError(f'{self.__class__} is not implemented.')
    
    def init_cache(self):
        raise NotImplementedError("init_cache() is not implemented.")
    
    def prev_update(self):
        raise NotImplementedError("prev_update() is not implemented.")
    
    def update_params(self):
        raise NotImplementedError("update_params() is not implemented.")
    
    def step(self):
        raise NotImplementedError("step() is not implemented.")
    
class Adam(Optimizer):
    def __init__(self, lr=0.001, decay=0, betas=(0.9, 0.999), epsilon=1e-8):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.epsilon = epsilon
        self.t = 1
        self.m = None
        self.v = None

    @classmethod
    def init_class(cls):
        pass

    def init_cache(self, n_params):
        self.m = [0] * n_params
        self.v = [0] * n_params

    def prev_update(self):
        if self.decay:
            self.current_lr = self.lr * (1 / (1 + self.t * self.decay))

    def update_params(self, gradient, i):
        self.m[i] = self.m[i] * self.beta_1 + (1 - self.beta_1) * gradient
        self.v[i] = self.v[i] * self.beta_2 + (1 - self.beta_2) * gradient**2
        m_h = self.m[i] / (1 - self.beta_1**self.t)
        v_h = self.v[i] / (1 - self.beta_2**self.t)
        return self.current_lr * (m_h / (np.sqrt(v_h) + self.epsilon))
    
    def step(self):
        self.t += 1