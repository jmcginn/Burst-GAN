
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose
from keras.layers.convolutional import UpSampling2D,UpSampling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D,AveragePooling1D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from functools import partial

import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random
from scipy import signal
import pickle
import sys
from math import *
import lal
from pylal import inject
import numpy as np

# Define gloabal parameters
outpath = "/data/public_html/2107829/MSci_Project/generated_samples/gen%05d.png"
batch_outpath = "/data/public_html/2107829/MSci_Project/batch_samples/batch%05d.png"
loss_path = "/data/public_html/2107829/MSci_Project/"
max_itr = 2*1000                                                # max no of steps/iterations
sample_rate = 1024
s_type = 'MIX'                                                   # Type of signal to generate 'SG' = sine gaussian, 'RD' = ringdonw, 'WNB' = white noise burst, 'MIX' = random mix of the three.
noise_switch = 'off'                                            # 'off' = no noise added to signal 'on' = noise added
#A = 1.0                                                         # Amplitude                        
t = np.linspace(0,1,sample_rate)                                # time vector in a 1 second range
phi = 2*np.pi
batch_size = 25
gps_time = 1215265166.000                                       # arbitrarily chosen
hp_d = 32							# hyper perameter that determines model size/filters
hp_g = 32
def clean_signals(s_type):
    '''
    This generates Sine Gaussian, ringdown and white noise burst signals depending on what you want.
    Also outputs the same signal but with a time delay corresponding to the light travel time between
    two detectors.
    '''
    offset = timeDelay(gps_time,'H1','L1') 

    if s_type == 'SG':
        A = 0.5                                                     # Amplitude
        f_0 = np.random.uniform(30,50)                                                   # frequency 
        t_0 =  np.random.uniform(0.2,0.8)                                              # starting epoch
        tau = np.random.uniform(1.0/60.0,1.0/15.0)                                       # decay constant 
        h = A * np.exp(-1.0*(t-t_0)**2/(tau**2))*np.sin(2*np.pi*f_0*(t-t_0) + phi)       # signal
        offset = timeDelay(gps_time,'H1','L1')                                           # time offset between two detectors 
        t_0 = t_0 + offset                                                               # update on epoch to run through same sine gaussian function
        h_offset = A * np.exp(-1.0*((t-t_0)/(tau))**2)*np.sin(2*np.pi*f_0*(t-t_0) + phi) # offset signal

    if s_type == 'RD':
        A = 0.5
        f_0 = np.random.uniform(30,50)
        t_0 = np.random.uniform(0.1,0.7)
        tau = np.random.uniform(0.02,0.1)
        offset = timeDelay(gps_time,'H1','L1')
        h = A * np.exp(-1.0*((t-t_0)/(tau)))*np.sin(2*np.pi*f_0*(t-t_0) + phi)
        h = ((t-t_0)>0)*h         # shifts signal by  making the start time non zero
        t_0 = t_0 + offset
        h_offset = A * np.exp(-1.0*((t-t_0)/(tau)))*np.sin(2*np.pi*f_0*(t-t_0) + phi)
        h_offset = ((t-t_0>0))*h_offset
    
    if s_type == 'WNB':
        '''
        unorthodox way of generating white noise bursts and  must be changed. Creates a string of zeros, picks a random point along that string to add
        gaussian noise to then resizes according to the sample rate. The offset is added but then passed as an integer to insert as it needs an integer
        for the index.We loose information when rounding down to the integer.
        '''
        flatline = np.zeros(sample_rate)
        t_0 = np.random.randint(0,sample_rate/2)
        noise = np.random.normal(0,np.random.uniform(0.1,0.2),(170))
        h = np.insert(flatline,t_0,noise)
        h.resize(sample_rate)
        offset = int(t_0 + (sample_rate * timeDelay(gps_time,'H1','L1')))

        h_offset = np.insert(flatline,offset,noise)
        h_offset.resize(sample_rate)
    return h, h_offset

def mix_sigs():
    '''
    Simple function to feed clean_signals a random choice of signal to generate
    '''
    func_list = ['SG','RD','WNB']
    function = random.choice(func_list)
    return function

def noisy(noise_typ,image,noise_switch):
    '''
    Adds gaussian noise to a signal
    '''
    if noise_switch == 'off':
        return image

    if noise_typ == "gauss" and noise_switch == 'on':
        t_series = image.shape
        mean = 0
        sigma = 0.2
        gauss = np.random.normal(mean,sigma,(t_series))
        gauss = gauss.reshape(t_series)
        noisy = image + gauss
    return noisy

def sig_gen(batch_size,s_type):

    '''
    Takes in a choice of signal type and batch size. Computes 1000 noisy signals and
    sets up an array of size: batch_size. Then reshapes for GAN training.. 
    '''

    signals = []
    signals_offset = []
    t = np.linspace(0,1,sample_rate)

    if s_type == 'MIX':
        for i in range(batch_size):
            h, h_offset = clean_signals(mix_sigs())
            signals.append(noisy('gauss',h,noise_switch))
            signals_offset.append(noisy('gauss',h_offset,noise_switch))

    else:
        for i in range(batch_size):
            h, h_offset = clean_signals(s_type)
            signals.append(noisy('gauss',h,noise_switch))
            signals_offset.append(noisy('gauss',h_offset,noise_switch))
    return np.array(signals).reshape(batch_size,sample_rate),np.array(signals_offset).reshape(batch_size,sample_rate)

def combine_gen_images(signals, nrows=5, ncols=5):
    '''
    Makes a 5x5 tile figure from a sample of a given signal array.
    '''
    signals = np.dstack(np.stack((signals),axis=1))
    fig, axs = plt.subplots(nrows, ncols, figsize=(20,10))
    fig.suptitle("Generated Samples", fontsize=18)
    fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical',fontsize=20)
    axs = axs.ravel()

    for i in range(nrows*ncols):
        axs[i].plot(t,signals[i][0],'b')# plots the signals
        axs[i].plot(t,signals[i][1],'g')# plots the offset
        axs[i].set_ylim([-1,1])
    return fig

def combine_batch_images(signals, nrows=5, ncols=5):
    '''
    Makes a 5x5 tile figure from a sample of a given signal array.
    '''
    signals = np.dstack(np.stack((signals),axis=1))
    fig, axs = plt.subplots(nrows, ncols, figsize=(20,10))
    fig.suptitle("Batch Samples", fontsize=18)
    fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical',fontsize=20)
    axs = axs.ravel()

    for i in range(nrows*ncols):
        axs[i].plot(t,signals[i][0],'b')# plots the signals
        axs[i].plot(t,signals[i][1],'g')# plots the offset
        axs[i].set_ylim([-1,1])
    return fig

def timeDelay( gpsTime, det1, det2 ):
  """
  Computes the time delay between two detectors dependant on a theoretical signals sky orientation.
  gpsTime is kept constant, however, right ascension and declination are random. Declination is 
  passed through arcsin function to remove over representation of signals in detectors blind spots.
  A positive time delay means the GW arrives first at 'det2', then at 'det1'.
  """

  ra_rad = np.random.uniform(0,2*pi)                                       # right ascension in rads
  de_rad = np.arcsin(np.random.uniform(-1,1))                             # declination

  if det1 == det2:
    return 0.0

  gps = lal.LIGOTimeGPS( gpsTime )

  # create detector-name map
  detMap = {'H1': 'LHO_4k', 'H2': 'LHO_2k', 'L1': 'LLO_4k',
            'G1': 'GEO_600', 'V1': 'VIRGO', 'T1': 'TAMA_300'}

  x1 = inject.cached_detector[detMap[det1]].location
  x2 = inject.cached_detector[detMap[det2]].location
  timedelay = lal.ArrivalTimeDiff(list(x1), list(x2), ra_rad, de_rad, gps)

  return timedelay

def plot_losses(losses,filename,logscale=False,legend=None):
    """
    Make loss and accuracy plots and output to file.
    Plot with x and y log-axes is desired
    """

    # plot losses
    fig = plt.figure()
    losses = np.array(losses)
    ax1 = fig.add_subplot(211)
    ax1.plot(losses[:,0],'g')
    if losses.shape[1]>2:
        ax1.plot(losses[:,2],'r')
    ax1.set_ylabel(r'loss')
    if legend is not None:
        ax1.legend(legend,loc='upper left')

    # plot accuracies
    #ax2 = fig.add_subplot(212)
    #ax2.plot(logit(losses[:,1]),'g')
    #if losses.shape[1]>3:
     #   ax2.plot(logit(losses[:,3]),'r')
    #ax2.set_yticks(logit([0.001,0.01,0.1,0.5,0.9,0.99,0.999]))
    #ax2.set_yticklabels(['0.001','0.01','0.1','0.5','0.9','0.99','0.999'])
    #ax2.set_xlabel(r'iterations')
    #ax2.set_ylabel(r'accuracy')
    if logscale==True:
        ax1.set_xscale("symlog", nonposx='clip')
        #ax1.set_xlim(0.001,10)
        ax2.set_xscale("symlog", nonposx='clip')
        ax1.set_yscale("symlog", nonposx='clip')
    plt.savefig(filename)
    plt.close('all')

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((batch_size, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        self.img_rows = 1024
        
        self.channels = 2
        self.img_shape = (self.img_rows, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = Adam(lr=1e-4,beta_1=0.5,beta_2=0.9)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(100,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10],metrics=['accuracy'])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
	pad = 'same'
	ks = 25
	act = 'relu'
	
        model.add(Dense(256*hp_g, input_dim=self.latent_dim))
	model.add(Activation(act))
	model.add(Reshape((1,16,16*hp_g)))

        model.add(Conv2DTranspose(8*hp_g, strides = (1,4), kernel_size=(1,ks), padding=pad))
	#model.add(BatchNormalization(momentum=0.8))
	model.add(Activation(act))

        model.add(Conv2DTranspose(4*hp_g,strides=(1,4), kernel_size=(1,ks), padding=pad))
	#model.add(BatchNormalization(momentum=0.8))
	model.add(Activation(act))

        model.add(Conv2DTranspose(self.channels, strides = (1,4), kernel_size=(1,ks), padding=pad))
 	model.add(Activation("tanh"))
	model.add(Reshape((1024,2)))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()
	pad = "same"
	ks = 25
	act = 'elu'

        model.add(Conv1D(16*hp_d, kernel_size=ks, strides=4, input_shape=self.img_shape, padding=pad))
	model.add(LeakyReLU(alpha=0.2))
	#model.add(Dropout(0.25))

        model.add(Conv1D(8*hp_d, kernel_size=ks, strides=4, padding=pad))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(BatchNormalization(momentum=0.8))
	#model.add(Dropout(0.25))

        model.add(Conv1D(4*hp_d, kernel_size=ks, strides=4, padding=pad))
        #model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25)) 
	model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
       # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
       # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
	
	#valid_d = [np.random.uniform(-0.9,-1) for i in range(batch_size)]
        #fake_d = [np.random.uniform(0.9,1) for i in range(batch_size)]
        losses = []
        for epoch in range(epochs):
            batch_signal,batch_signal_delay = sig_gen(batch_size,s_type) # arrays of signals and their offsets
            X_train = np.stack((batch_signal,batch_signal_delay), axis=2)

	    for _ in range(self.n_critic):

		# ---------------------
		#  Train Discriminator
		# ---------------------

		# Select a random batch of images
		#idx = np.random.randint(0, X_train.shape[0], batch_size)
		#imgs = X_train[idx]
		# Sample generator input
		noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)
		# Train the critic
		d_loss = self.critic_model.train_on_batch([X_train, noise],
								[valid, fake, dummy])


	    # ---------------------
	    #  Train Generator
	    # ---------------------

	    g_loss = self.generator_model.train_on_batch(noise, valid)
	    losses.append([g_loss,g_loss,d_loss[0],d_loss[0]])

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
	    # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
		print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
 		noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)
		generated_signals = self.generator.predict(noise)
                gen_image = combine_gen_images(generated_signals)
		gen_image.savefig(outpath % epoch)
		batch_image = combine_batch_images(X_train)
                batch_image.savefig(batch_outpath % epoch)
		plot_losses(losses,'%s/losses.png' % loss_path,logscale=False, legend=['Generator','Discriminator'])
		#d_loss_train = self.critic_model.test_on_batch([X_train],[valid,dummy])
		#print(d_loss_train)



if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=batch_size, sample_interval=100)
