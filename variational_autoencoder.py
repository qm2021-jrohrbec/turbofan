import numpy as np
import tensorflow as tf
from keras.models import Model
from keras import layers
import typing
from typing import Optional

class Autoencoder(Model):
    '''
    Basic Autoencoder.

    Hidden units user specific or of size h1 = Z(input * 1.2), h2 = Z(h1 * 0.5),
    h3 = Z(h3 * 0.5), given input size.

    - Input x -Encoder-> z -Decoder-> x*
    - Optimization: min Loss(x,x*)

    Usage:
    ae = Autoencoder(input_dim)
    ae.compile(optimizer,loss)
    ae.fit(x,x,...)
    '''

    def __init__(self, input_dim: int, hidden_dims: Optional[list[int]] = None, activation: str = 'relu'):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim

        if hidden_dims == None:
            h1 = int(self.input_dim * 1.2)
            h2 = int(h1 * 0.5)
            self.hidden_dims = [h1, h2]
            self.code_dim = int(h2 * 0.5)
        
        # Encoder, Decoder
        encoder_layers = []
        decoder_layers = []
        debth = len(self.hidden_dims)
        for i in range(1, debth):
            encoder_layers.append(layers.Dense(self.hidden_dims[i], activation = activation))
            decoder_layers.append(layers.Dense(self.hidden_dims[debth - 1 - i], activation = activation))
        self.encoder = tf.keras.Sequential(encoder_layers)
        self.code = layers.Dense(self.code_dim)
        self.decoder = tf.keras.Sequential(decoder_layers)
        self.output_layer = layers.Dense(self.input_dim)

    def call(self, x):
        encoded = self.encoder(x)
        code = self.code(encoded)
        decoded = self.decoder(code)
        return self.output_layer(decoded)

class VariationalAutoencoder(Model):
    '''
    Basic 'Variational' Autoencoder (no evidence lower bound implemented here)

    Gaussian latent distribution z ~ N(mu, var). Hidden units user specific or of size 
    h1 = Z(input * 1.2), h2 = Z(h1 * 0.5), h3 = Z(h3 * 0.5), given input size.
    
    - Prior distribution N(0,1)
    - Input x -Encoder-> mu, log(var) -Reparametrization-> z ~ N(mu, var) -Decoder-> x*
    - Optimization: min Loss(x,x*)

    Usage:
    ae = Autoencoder(input_dim)
    ae.compile(optimizer,loss)
    ae.fit(x,x,...)
    '''

    def __init__(self, input_dim: int, hidden_dims: Optional[list[int]] = None, activation: str = 'relu'):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim

        if hidden_dims == None:
            h1 = int(self.input_dim * 1.2)
            h2 = int(h1 * 0.5)
            self.hidden_dims = [h1, h2]
            self.code_dim = int(h2 * 0.5)
        
        # Encoder, Decoder
        encoder_layers = []
        decoder_layers = []
        debth = len(self.hidden_dims)
        for i in range(1, debth):
            encoder_layers.append(layers.Dense(self.hidden_dims[i], activation = activation))
            decoder_layers.append(layers.Dense(self.hidden_dims[debth - 1 - i], activation = activation))
        self.encoder = tf.keras.Sequential(encoder_layers)
        self.code = layers.Dense(self.code_dim + self.code_dim)
        self.decoder = tf.keras.Sequential(decoder_layers)
        self.output_layer = layers.Dense(self.input_dim)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.code))
        return self.decode(eps)

    def encode(self, x):
        encoded = self.encoder(x)
        mean, logvar = tf.split(self.code(encoded), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        '''
        eps ~ N(0,1) -> eps * sqrt(var) + mean ~ N(mean, var)
        '''
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.keras.backend.random_normal(shape=(batch, dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        decoded = self.decoder(z)
        return self.output_layer(decoded)

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

class VariationalAutoencoderELBOMC(Model):
    '''
    Variational Autoencoder.

    Gaussian latent distribution z ~ N(mu, var). Hidden units user specific or of size 
    h1 = Z(input * 1.2), h2 = Z(h1 * 0.5), h3 = Z(h3 * 0.5), given input size.
    
    - Prior distribution N(0,1)
    - Input x -Encoder-> mu, log(var) -Reparametrization-> z ~ N(mu, var) -Decoder-> x*
    - Optimization: max ELBO by using Monte Carlo

    Usage:
    ae = Autoencoder(input_dim)
    ae.compile(optimizer,loss)
    ae.fit(x,epochs,batch_size)
    '''

    def __init__(self, input_dim: int, hidden_dims: Optional[list[int]] = None, activation: str = 'relu', **kwargs):
        super(VariationalAutoencoderELBOMC, self).__init__(**kwargs)
        self.input_dim = input_dim

        if hidden_dims == None:
            h1 = int(self.input_dim * 1.2)
            h2 = int(h1 * 0.5)
            self.hidden_dims = [h1, h2]
            self.code_dim = int(h2 * 0.5)
        
        # Encoder, Decoder
        encoder_layers = []
        decoder_layers = []
        debth = len(self.hidden_dims)
        for i in range(1, debth):
            encoder_layers.append(layers.Dense(self.hidden_dims[i], activation = activation))
            decoder_layers.append(layers.Dense(self.hidden_dims[debth - 1 - i], activation = activation))
        self.encoder = tf.keras.Sequential(encoder_layers)
        self.code = layers.Dense(self.code_dim + self.code_dim)
        self.decoder = tf.keras.Sequential(decoder_layers)
        self.output_layer = layers.Dense(self.input_dim)

        self.loss_mean = tf.keras.metrics.Mean(name="loss")
        self.logpxz_mean = tf.keras.metrics.Mean(name="logpxz")
        self.logpz_mean = tf.keras.metrics.Mean(name="logpz")
        self.logqzx_mean = tf.keras.metrics.Mean(name="logqzx")

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.code))
        return self.decode(eps)

    def encode(self, x):
        encoded = self.encoder(x)
        mean, logvar = tf.split(self.code(encoded), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        '''
        eps ~ N(0,1) -> eps * sqrt(var) + mean ~ N(mean, var)
        '''
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.keras.backend.random_normal(shape=(batch, dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        decoded = self.decoder(z)
        return self.output_layer(decoded)

    def predict(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

    def log_pdf_function(self, sample, mean, logvar):
        '''
        calculates log of probability density function,
        log p(sample), sample ~ Normal(mean, exp(logvar)

        log((1/sqrt(var*2*pi))*exp(-.5((sample-mean)**2/var))
            = -.5 ( (sample-mean)**2 ) / var - .5 logvar - .5 log2pi
        '''
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=1)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            reconstruction = self.decode(z)

            logpxz = -tf.keras.losses.binary_crossentropy(x, reconstruction)
            logpz = self.log_pdf_function(z, 0., 0.)
            logqzx = self.log_pdf_function(z, mean, logvar)

            loss = -tf.reduce_mean(logpxz + logpz - logqzx)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.loss_mean.update_state(loss),
        self.logpxz_mean.update_state(logpxz),
        self.logpz_mean.update_state(logpz),
        self.logqzx_mean.update_state(logqzx)
        return {
            "loss": self.loss_mean.result(),
            "logpxz": self.logpxz_mean.result(),
            "logpz": self.logpz_mean.result(),
            "logqzx": self.logqzx_mean.result()
        }

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

class VariationalAutoencoderELBOKL(Model):
    '''
    Variational Autoencoder.

    Gaussian latent distribution z ~ N(mu, var). Hidden units user specific or of size 
    h1 = Z(input * 1.2), h2 = Z(h1 * 0.5), h3 = Z(h3 * 0.5), given input size.
    
    - Prior distribution N(0,1)
    - Input x -Encoder-> mu, log(var) -Reparametrization-> z ~ N(mu, var) -Decoder-> x*
    - Optimization: max ELBO by using Cross Entropy and KL Divergence

    Usage:
    ae = Autoencoder(input_dim)
    ae.compile(optimizer,loss)
    ae.fit(x,epochs,batch_size)
    '''

    def __init__(self, input_dim: int, hidden_dims: Optional[list[int]] = None, activation: str = 'relu', **kwargs):
        super(VariationalAutoencoderELBOMC, self).__init__(**kwargs)
        self.input_dim = input_dim

        if hidden_dims == None:
            h1 = int(self.input_dim * 1.2)
            h2 = int(h1 * 0.5)
            self.hidden_dims = [h1, h2]
            self.code_dim = int(h2 * 0.5)
        
        # Encoder, Decoder
        encoder_layers = []
        decoder_layers = []
        debth = len(self.hidden_dims)
        for i in range(1, debth):
            encoder_layers.append(layers.Dense(self.hidden_dims[i], activation = activation))
            decoder_layers.append(layers.Dense(self.hidden_dims[debth - 1 - i], activation = activation))
        self.encoder = tf.keras.Sequential(encoder_layers)
        self.code = layers.Dense(self.code_dim + self.code_dim)
        self.decoder = tf.keras.Sequential(decoder_layers)
        self.output_layer = layers.Dense(self.input_dim)

        self.loss_mean = tf.keras.metrics.Mean(name="loss")
        self.ce_mean = tf.keras.metrics.Mean(name="cross entropy")
        self.kl_mean = tf.keras.metrics.Mean(name="kl divergence")

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.code))
        return self.decode(eps)

    def encode(self, x):
        encoded = self.encoder(x)
        mean, logvar = tf.split(self.code(encoded), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        '''
        eps ~ N(0,1) -> eps * sqrt(var) + mean ~ N(mean, var)
        '''
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.keras.backend.random_normal(shape=(batch, dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        decoded = self.decoder(z)
        return self.output_layer(decoded)

    def predict(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            reconstruction = self.decode(z)

            cross_entropy = tf.keras.losses.binary_crossentropy(x, reconstruction)
            kl = -.5 * tf.reduce_sum(1 + logvar - mean * mean - tf.exp(logvar), axis=1)
            loss = tf.reduce_mean(cross_entropy + kl)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.loss_mean.update_state(loss),
        self.ce_mean.update_state(cross_entropy),
        self.kl_mean.update_state(kl),
        return {
            "loss": self.loss_mean.result(),
            "cross entropy": self.ce_mean.result(),
            "kl div": self.kl_mean.result(),
        }
    
    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

class VariationalAutoencoderELBO_MSE_KL(Model):
    '''
    Variational Autoencoder.

    Gaussian latent distribution z ~ N(mu, var). Hidden units user specific or of size 
    h1 = Z(input * 1.2), h2 = Z(h1 * 0.5), h3 = Z(h3 * 0.5), given input size.
    
    - Prior distribution N(0,1)
    - Input x -Encoder-> mu, log(var) -Reparametrization-> z ~ N(mu, var) -Decoder-> x*
    - Optimization: max ELBO by using Cross Entropy and KL Divergence

    Usage:
    ae = Autoencoder(input_dim)
    ae.compile(optimizer,loss)
    ae.fit(x,epochs,batch_size)
    '''

    def __init__(self, input_dim: int, hidden_dims: Optional[list[int]] = None, activation: str = 'relu', **kwargs):
        super(VariationalAutoencoderELBO_MSE_KL, self).__init__(**kwargs)
        self.input_dim = input_dim

        if hidden_dims == None:
            h1 = int(self.input_dim * 1.2)
            h2 = int(h1 * 0.5)
            self.hidden_dims = [h1, h2]
            self.code_dim = int(h2 * 0.5)
        
        # Encoder, Decoder
        encoder_layers = []
        decoder_layers = []
        debth = len(self.hidden_dims)
        for i in range(1, debth):
            encoder_layers.append(layers.Dense(self.hidden_dims[i], activation = activation))
            decoder_layers.append(layers.Dense(self.hidden_dims[debth - 1 - i], activation = activation))
        self.encoder = tf.keras.Sequential(encoder_layers)
        self.code = layers.Dense(self.code_dim + self.code_dim)
        self.decoder = tf.keras.Sequential(decoder_layers)
        self.output_layer = layers.Dense(self.input_dim)

        self.loss_mean = tf.keras.metrics.Mean(name="loss")
        self.mse_mean = tf.keras.metrics.Mean(name="mse loss")
        self.kl_mean = tf.keras.metrics.Mean(name="kl divergence")

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.code))
        return self.decode(eps)

    def encode(self, x):
        encoded = self.encoder(x)
        mean, logvar = tf.split(self.code(encoded), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        '''
        eps ~ N(0,1) -> eps * sqrt(var) + mean ~ N(mean, var)
        '''
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.keras.backend.random_normal(shape=(batch, dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        decoded = self.decoder(z)
        return self.output_layer(decoded)

    def predict(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            reconstruction = self.decode(z)

            xdx= tf.metrics.mean_squared_error(x, reconstruction)
            kl = -.5 * tf.reduce_sum(1 + logvar - mean * mean - tf.exp(logvar), axis=1)
            loss = tf.reduce_mean(xdx + kl)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.loss_mean.update_state(loss),
        self.mse_mean.update_state(xdx),
        self.kl_mean.update_state(kl),
        return {
            "loss": self.loss_mean.result(),
            "mse loss": self.mse_mean.result(),
            "kl div": self.kl_mean.result(),
        }