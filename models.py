import numpy as np
import pickle
import time

from functions import *
from optimizers import *
from utils import *
from settings import DTYPE

class Model:
    count = 0
    def __init__(self, name=None):
        if name:
            self.name = name
        else:
            self.name = f'{self.__class__.__name__}_{__class__.count}'

        __class__.count += 1

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
class Generator(Model):
    def __init__(self, z_dim: int, n_classes: int, out_dim: int, dtype: np.dtype, name: str=None):
        super().__init__(name)

        self.dtype = dtype
        self.z_dim = z_dim
        self.n_classes = n_classes

        self.n_INPUTS = z_dim + n_classes
        self.n_hidden0 = 256
        self.n_hidden1 = 512
        self.n_OUTPUTS = out_dim

        self.init_params()

        self.params = [self.W0, self.b0,
                       self.W1, self.b1,
                       self.W2, self.b2]
        
        self.n_params = len(self.params)

    def init_params(self):
        self.W0 = np.random.randn(self.n_INPUTS, self.n_hidden0).astype(self.dtype) * np.sqrt(2/self.n_INPUTS)
        self.b0 = np.zeros((1, self.n_hidden0)).astype(self.dtype)

        self.W1 = np.random.randn(self.n_hidden0, self.n_hidden1).astype(self.dtype) * np.sqrt(2/self.n_hidden0)
        self.b1 = np.zeros((1, self.n_hidden1)).astype(self.dtype)

        self.W2 = np.random.randn(self.n_hidden1, self.n_OUTPUTS).astype(self.dtype) * np.sqrt(1/self.n_hidden1)
        self.b2 = np.zeros((1, self.n_OUTPUTS)).astype(self.dtype)

    def compile(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.optimizer.init_cache(self.n_params)

    def update_params(self, gradients):
        self.optimizer.prev_update()
        for i in range(self.n_params):
            self.params[i] -= self.optimizer.update_params(gradients[i], i)
        self.optimizer.step()

    def make_random_tokens(self, total=1):
        z = np.random.standard_normal(size=(total, self.z_dim))
        token = one_hot(np.random.randint(low=0, high=self.n_classes, size=total), self.n_classes)
        return z, token
    
    def generate(self, tokens):
        return self.forward(np.random.standard_normal(size=(tokens.shape[0], self.z_dim)), tokens)
    
    def generate_random(self, total=1):
        return self.forward(*self.make_random_tokens(total=total))

    def forward(self, z, token):
        self.w = np.concatenate([z, token], axis=1)

        self.z0 = self.w @ self.W0 + self.b0
        self.a0 = ReLU(self.z0)

        self.z1 = self.a0 @ self.W1 + self.b1
        self.a1 = ReLU(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = tanh(self.z2)

        return self.a2
    
    def backward(self, deltas=None, learn=True):
        m = deltas.shape[0]
        deltas = deltas[:, :self.n_OUTPUTS]
        deltas *= tanh(self.a2, derv=True)

        dW2 = self.a1.T @ deltas / m
        db2 = np.sum(deltas, axis=0, keepdims=True) / m

        dz1 = deltas @ self.W2.T * ReLU(self.z1, derv=True)

        dW1 = self.a0.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        dz0 = dz1 @ self.W1.T * ReLU(self.z0, derv=True)

        dW0 = self.w.T @ dz0 / m
        db0 = np.sum(dz0, axis=0, keepdims=True) / m

        gradients = (dW0, db0,
                     dW1, db1,
                     dW2, db2)
        
        if learn:
            self.update_params(gradients)

class Discriminator(Model):
    def __init__(self, inp_dim: int, n_classes: int, dtype: np.dtype, name: str=None):
        super().__init__(name)

        self.dtype = dtype
        self.inp_dim = inp_dim
        self.n_classes = n_classes

        self.n_INPUTS = inp_dim + n_classes
        self.n_hidden0 = 512
        self.n_hidden1 = 256
        self.n_OUTPUTS = 1

        self.init_params()

        self.params = [self.W0, self.b0,
                       self.W1, self.b1,
                       self.W2, self.b2]
        
        self.n_params = len(self.params)

    def init_params(self):
        self.W0 = np.random.randn(self.n_INPUTS, self.n_hidden0).astype(self.dtype) * np.sqrt(2/self.n_INPUTS)
        self.b0 = np.zeros((1, self.n_hidden0)).astype(self.dtype)

        self.W1 = np.random.randn(self.n_hidden0, self.n_hidden1).astype(self.dtype) * np.sqrt(2/self.n_hidden0)
        self.b1 = np.zeros((1, self.n_hidden1)).astype(self.dtype)

        self.W2 = np.random.randn(self.n_hidden1, self.n_OUTPUTS).astype(self.dtype) * np.sqrt(1/self.n_hidden1)
        self.b2 = np.zeros((1, self.n_OUTPUTS)).astype(self.dtype)

    def compile(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.optimizer.init_cache(self.n_params)

    def update_params(self, gradients):
        self.optimizer.prev_update()
        for i in range(self.n_params):
            self.params[i] -= self.optimizer.update_params(gradients[i], i)
        self.optimizer.step()

    def forward(self, x, token):
        self.inp = np.concatenate([x, token], axis=1)

        self.z0 = self.inp @ self.W0 + self.b0
        self.a0 = LeakyReLU(self.z0)

        self.z1 = self.a0 @ self.W1 + self.b1
        self.a1 = LeakyReLU(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2
    
    def backward(self, y=None, learn=True):
        dL = BinaryCrossEntropyLoss(y, self.a2, derv=True)
        dL *= sigmoid(self.a2, derv=True)
        m = dL.shape[0]

        dW2 = self.a1.T @ dL / m
        db2 = np.sum(dL, axis=0, keepdims=True) / m

        dz1 = dL @ self.W2.T * LeakyReLU(self.z1, derv=True)

        dW1 = self.a0.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        dz0 = dz1 @ self.W1.T * LeakyReLU(self.z0, derv=True)

        dW0 = self.inp.T @ dz0 / m
        db0 = np.sum(dz0, axis=0, keepdims=True) / m

        deltas = dz0 @ self.W0.T

        gradients = (dW0, db0,
                     dW1, db1,
                     dW2, db2)
        
        if learn:
            self.update_params(gradients)

        return deltas
    
    def teach(self, y=None): # This functions is to teach Generator even faster.
        dL = BinaryCrossEntropyLoss(y, self.a2, derv=True) * sigmoid(self.a2, derv=True)
        
        dz1 = dL @ self.W2.T * LeakyReLU(self.z1, derv=True)

        dz0 = dz1 @ self.W1.T * LeakyReLU(self.z0, derv=True)

        deltas = dz0 @ self.W0.T

        return deltas

class NeuralLayer(Model): # This class is going to be use for autoencoder (cleaner of noise, reshaper, ...)
    def __init__(self, inp_dim: int, out_dim: int, activation, dtype: np.dtype, name: str=None):
        super().__init__(name)
        
        self.dtype = dtype
        self.activation = activation
        self.init_params(inp_dim, out_dim)

    def init_params(self, inp_dim, out_dim):
        self.W = np.random.randn(inp_dim, out_dim).astype(self.dtype) * np.sqrt(2/inp_dim)
        self.b = np.zeros((1, out_dim)).astype(self.dtype)

    def forward(self, x):
        self.x = x

        self.z0 = x @ self.W + self.b
        self.a0 = self.activation(self.z0)
        return self.a0
    
    def backward(self, dout):
        dout *= self.activation(self.a0, derv=True)

        dW = self.x.T @ dout / dout.shape[0]
        db = np.sum(dout, axis=0, keepdims=True) / dout.shape[0]

        return dout @ self.W.T, (dW, db)

class AutoEncoder(Model):
    def __init__(self, encoder: NeuralLayer, decoder: NeuralLayer, name: str=None):
        super().__init__(name)

        self.encoder = encoder
        self.decoder = decoder

        self.params = [self.encoder.W, self.encoder.b,
                       self.decoder.W, self.decoder.b]
        
        self.n_params = len(self.params)

    def compile(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.optimizer.init_cache(self.n_params) # weights + biases of encoder and decoder

    def update_params(self, gradients):
        self.optimizer.prev_update()
        for i in range(self.n_params):
            self.params[i] -= self.optimizer.update_params(gradients[i], i)
        self.optimizer.step()

    def forward(self, x):
        self.x = x

        self.z = self.encoder.forward(x)
        self.reconstruction = self.decoder.forward(self.z)

        return self.reconstruction, self.z
    
    def backward(self, y=None, learn=True):
        dout = BinaryCrossEntropyLoss(y, self.reconstruction, derv=True)
        dout, (decdW, decdb) = self.decoder.backward(dout)
        _, (encdW, encdb) = self.encoder.backward(dout)

        if learn:
            self.update_params((decdW, decdb, encdW, encdb))

    def save_layers(self, path_encoder: str, path_decoder: str):
        self.encoder.save(path_encoder)
        self.decoder.save(path_decoder)

    @classmethod
    def load_models(cls, path_encoder: str, path_decoder: str):
        encoder = cls.load(path_encoder)
        decoder = cls.load(path_decoder)
        return cls(encoder, decoder)

class ConditionalGAN(Model):
    def __init__(self, generator: Generator, discriminator: Discriminator, name: str=None):
        super().__init__(name)

        self.generator = generator
        self.discriminator = discriminator
        self.total_epochs = 0
        self.seed = None

    def make_grafical_follow(self, num_images=32, imgs_path:str=None):
        self.seed = self.generator.make_random_tokens(total=num_images)
        self.imgs_path = imgs_path

    def compile(self, optimizer_generator: Optimizer, optimizer_discriminator: Optimizer):
        self.generator.compile(optimizer_generator)
        self.discriminator.compile(optimizer_discriminator)

    def save_models(self, path_gen: str, path_disc: str):
        self.generator.save(path_gen)
        self.discriminator.save(path_disc)
    
    @classmethod
    def load_models(cls, path_gen: str, path_disc: str):
        generator = cls.load(path_gen)
        discriminator = cls.load(path_disc)
        return cls(generator, discriminator)
    
    def train(self, x=None, y=None, epochs=10, batch_size=256, d_steps=1, g_steps=1, alpha=1, beta=1, verbose=True, debug=False):
        half_batch = batch_size // 2

        for ep in range(1, epochs+1):
            start = time.time()

            for b in range(0, x.shape[0], half_batch):
                real_images = x[b:b+half_batch]
                real_tokens = y[b:b+half_batch]

                for _ in range(d_steps):
                    noise, fake_tokens = self.generator.make_random_tokens(total=real_images.shape[0])
                    fake_images = self.generator.forward(noise, fake_tokens)
                    
                    tokens = np.concatenate([real_tokens, fake_tokens], axis=0)
                    images = np.concatenate([real_images, fake_images], axis=0)
                    labels = np.concatenate([np.ones((real_images.shape[0], 1)) * alpha,
                                            np.zeros((real_images.shape[0], 1))], axis=0)

                    predictions = self.discriminator.forward(images, tokens)
                    
                    D_loss = BinaryCrossEntropyLoss(labels, predictions)

                    self.discriminator.backward(labels, learn=True)

                for _ in range(g_steps):
                    noise, fake_tokens = self.generator.make_random_tokens(total=real_images.shape[0]*2)
                    fake_images = self.generator.forward(noise, fake_tokens)

                    predictions = self.discriminator.forward(fake_images, fake_tokens)

                    labels = np.ones((noise.shape[0], 1)) * beta

                    G_loss = BinaryCrossEntropyLoss(labels, predictions)

                    deltas = self.discriminator.teach(labels)
                    self.generator.backward(deltas, learn=True)

                if debug:
                    print(f'Generator Loss: {G_loss:.4f}, Discriminator Loss: {D_loss:.4f}')

            if verbose:
                end = time.time()
                total_time = end-start
                generate_and_save_images(self.generator, self.seed, G_loss, D_loss, ep+self.total_epochs, self.imgs_path)
                print(f"{self.name} -> time: ({total_time:.2f} sec) G Loss: {G_loss:.4f}, D Loss: {D_loss:.4f}")
        
        self.total_epochs += epochs

class VAE(Model):
    def __init__(self, n_inputs: int, z_dim: int, neurons: int, n_outputs: int,
                 first_activation=LeakyReLU, second_activation=ReLU, last_activation=sigmoid,
                 dtype: np.dtype=DTYPE, name: str=None):
        super().__init__(name)
        self.dtype = dtype
        self._init_weights(n_inputs, z_dim, neurons, n_outputs)
        self.first_activation = first_activation
        self.second_activation = second_activation
        self.last_activation = last_activation

    def _init_weights(self, n_inputs, z_dim, neurons, n_outputs):
        # Encoder
        self.W0 = np.random.randn(n_inputs, neurons).astype(self.dtype) * np.sqrt(2/n_inputs)
        self.b0 = np.zeros((1, neurons)).astype(self.dtype)

        self.Wmu = np.random.randn(neurons, z_dim).astype(self.dtype) * np.sqrt(2/neurons)
        self.bmu = np.zeros((1, z_dim)).astype(self.dtype)
        self.Wlogvar = np.random.randn(neurons, z_dim).astype(self.dtype) * np.sqrt(2/neurons)
        self.blogvar = np.zeros((1, z_dim)).astype(self.dtype)

        # Decoder
        self.W1 = np.random.randn(z_dim, neurons).astype(self.dtype) * np.sqrt(2/z_dim)
        self.b1 = np.zeros((1, neurons)).astype(self.dtype)

        self.W2 = np.random.randn(neurons, n_outputs).astype(self.dtype) * np.sqrt(2/neurons)
        self.b2 = np.zeros((1, n_outputs)).astype(self.dtype)

        self.params = [self.W0, self.b0,
                       self.Wmu, self.bmu, self.Wlogvar, self.blogvar,
                       self.W1, self.b1,
                       self.W2, self.b2]
        
        self.n_params = len(self.params)

    def kl_div(self, mu, logvar, derv=False):
        if derv: return mu, -0.5 * (1 - np.exp(logvar))
        return -0.5 * np.mean(1 + logvar - np.power(mu, 2) - np.exp(logvar))

    def compile(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.optimizer.init_cache(self.n_params)

    def update_params(self, grads):
        self.optimizer.prev_update()
        for i in range(self.n_params):
            self.params[i] -= self.optimizer.update_params(grads[i], i)
        self.optimizer.step()

    def forward_encoder(self, x):
        self.z0 = x @ self.W0 + self.b0
        self.a0 = self.first_activation(self.z0)

        self.mue = self.a0 @ self.Wmu + self.bmu
        self.logvare = self.a0 @ self.Wlogvar + self.blogvar

        return self.mue, self.logvare
    
    def forward_decoder(self, z):
        self.z1 = z @ self.W1 + self.b1
        self.a1 = self.second_activation(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.last_activation(self.z2)

        return self.a2
    
    def forward(self, x):
        mue, logvare = self.forward_encoder(x)
        self.rand_sample = np.random.standard_normal(size=mue.shape)
        self.z = mue + np.exp(logvare * 0.5) * self.rand_sample
        reconstruction = self.forward_decoder(self.z)
        return reconstruction, mue, logvare
    
    def backward(self, x, outp, learn=True):
        batch_size = x.shape[0]

        dL = BinaryCrossEntropyLoss(x, outp, derv=True) / batch_size
        dL *= self.last_activation(outp, derv=True)

        dW2 = self.a1.T @ dL
        db2 = dL.sum(axis=0, keepdims=True)
        
        da1 = dL @ self.W2.T
        dz1 = da1 * self.second_activation(self.z1, derv=True)
        
        dW1 = self.z.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        
        dz = dz1 @ self.W1.T
        
        dlrelu = self.first_activation(self.z0, derv=True)
        
        # MU

        dmu = dz
        
        dWmu = self.a0.T @ dmu
        dbmu = dmu.sum(axis=0, keepdims=True)
        
        da0mu = dmu @ self.Wmu.T
        dz0mu = da0mu * dlrelu
        
        dW0mu = x.T @ dz0mu
        db0mu = dz0mu.sum(axis=0, keepdims=True)
        
        # LOGVAR

        dlogvar = dz * np.exp(self.logvare * 0.5) * 0.5 * self.rand_sample
        
        dWlogvar = self.a0.T @ dlogvar
        dblogvar = dlogvar.sum(axis=0, keepdims=True)
        
        da0logvar = dlogvar @ self.Wlogvar.T
        dz0logvar = da0logvar * dlrelu
        
        dW0logvar = x.T @ dz0logvar
        db0logvar = dz0logvar.sum(axis=0, keepdims=True)
        
        # KL DIV

        dKL_mu, dKL_logvar = self.kl_div(self.mue, self.logvare, derv=True)
        dKL_mu /= batch_size
        dKL_logvar /= batch_size
        
        # KL MU

        dKL_Wmu = self.a0.T @ dKL_mu
        dKL_bmu = dKL_mu.sum(axis=0, keepdims=True)
        
        dKL_a0mu = dKL_mu @ self.Wmu.T
        dKL_z0mu = dKL_a0mu * dlrelu
        
        dKL_W0mu = x.T @ dKL_z0mu
        dKL_b0mu = dKL_z0mu.sum(axis=0, keepdims=True)
        
        # KL LOGVAR

        dKL_Wlogvar = self.a0.T @ dKL_logvar
        dKL_blogvar = dKL_logvar.sum(axis=0, keepdims=True)
        
        dKL_a0logvar = dKL_logvar @ self.Wlogvar.T
        dKL_z0logvar = dKL_a0logvar * dlrelu
        
        dKL_W0logvar = x.T @ dKL_z0logvar
        dKL_b0logvar = dKL_z0logvar.sum(axis=0, keepdims=True)
        
        dW0 = dW0mu + dW0logvar + dKL_W0mu + dKL_W0logvar
        db0 = db0mu + db0logvar + dKL_b0mu + dKL_b0logvar
        
        dWmu += dKL_Wmu
        dbmu += dKL_bmu
        dWlogvar += dKL_Wlogvar
        dblogvar += dKL_blogvar
        
        grads = (dW0, db0,
                 dWmu, dbmu, dWlogvar, dblogvar,
                 dW1, db1,
                 dW2, db2)

        if learn:
            self.update_params(grads)
    
    def train(self, x, epochs=10, batch_size=32, print_every=0.1, shuffle=True, debug=False):
        total_batch = np.ceil(x.shape[0]/batch_size).astype(int)
        for ep in range(1, epochs+1):
            total_KL_loss = 0.0
            total_loss = 0.0

            if shuffle: np.random.shuffle(x)

            for batch in range(0, x.shape[0], batch_size):
                x_batch = x[batch:batch+batch_size]
                reconstruction, mue, logvare = self.forward(x_batch)
                self.backward(x=x_batch, outp=reconstruction, learn=True)
                KL_loss = self.kl_div(mue, logvare)
                loss = BinaryCrossEntropyLoss(x_batch, reconstruction) + KL_loss

                total_KL_loss += KL_loss
                total_loss += loss

                if debug:
                    print(f'Loss: {loss}, KL Loss: {KL_loss}')

            avg_loss = total_loss / total_batch
            avg_KL_loss = total_KL_loss / total_batch

            if ep % max(1, int(epochs * print_every)) == 0:
                print(f'Epoch: [{ep}/{epochs}] Avg Loss: {avg_loss:.4f} | Avg KL Loss: {avg_KL_loss:.4f}')

class DNN(Model):
    pass