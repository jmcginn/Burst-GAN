from keras.models import Sequential                             # linear stack of layers
from keras.layers import Dense                                  # define first hidden layer
from keras.layers import Reshape, Dropout
from keras.layers.core import Activation                        # maps output from -1 to 1
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D,UpSampling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU,ELU
from keras.layers.core import Flatten
from keras.optimizers import Adam                               # finds max/min/0's by differentiation
from scipy.special import logit, expit
import numpy as np                                              # of first order
from PIL import Image
import os
import glob
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import sys
from math import *
import lal
from pylal import inject

# stuff
n_colors = 1                                                    # for rgb colours, keep as 1 for times series, currently not compatible with images
filepath = './data/images/*.jpg'
outpath = "/data/public_html/2107829/Burst/SG_gan/gen_im/gen_var%05d.png"
batch_outpath = "/data/public_html/2107829/Burst/SG_gan/batch_im/batch_var%05d.png"
loss_path = "/data/public_html/2107829/Burst/SG_gan/"
max_itr = 5*1000                                                # max no of steps/iterations
sample_rate = 512
s_type = 'MIX'                                                   # Type of signal to generate 'SG' = sine gaussian, 'RD' = ringdonw, 'WNB' = white noise burst, 'MIX' = random mix of the three.
noise_switch = 'off'                                            # 'off' = no noise added to signal 'on' = noise added
A = 1.0                                                         # Amplitude                        
t = np.linspace(0,1,sample_rate)                                # time vector in a 1 second range
phi = 2*np.pi
batch_size = 64
gps_time = 1215265166.000                                       # arbitrarily chosen

def generator_model():
    """
    Takes random number as input and trains itself to generate required output
    """

    model = Sequential()
    act = 'tanh'
    momentum = 0.9
    dropout = 0.6

    # the first dense layer converts 100 random nubers into 1024 numbers and outputs
    # with tan activation in range -1 to 1
    model.add(Dense(1024, input_shape=(100,)))                  # only first layer needs input shape
    model.add(Activation(act))
    
    # the second dense layer expands to 12*16*16=32768 w/tanh Activation
    model.add(Dense(128 * 256 * 1))
    #model.add(BatchNormalization())
    model.add(Activation(act))
    

    # reshape into two 1D vectors  double the size in 2nd  dimension
    # apply 2dconvolution to output dimensions 64 over a convolved
    # window of height1/width 2
    model.add(Reshape((2, 128, 128)))
    model.add(UpSampling2D(size=(1,2)))
    model.add(Conv2D(64, (2,2), padding='same'))
    model.add(Activation(act))
    
    #model.add(UpSampling2D(size=(1, 2)))
    #model.add(Conv2D(32, (2,5), padding='same'))
    #model.add(BatchNormalization(momentum=momentum))
    #model.add(Activation(act)) 

    model.add(UpSampling2D(size=(1, 2)))
    model.add(Conv2D(n_colors, (2,2), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    act='tanh'
    momentum=0.9
    dropout = 0.4
   
    # apply 2Dconvolution with stride, tanh activation and double 2nd dimension 
    model.add(Conv2D(64, (2,2),strides=(1,1),input_shape=(2, sample_rate, n_colors), padding='same'))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(1,2)))
    #model.add(Dropout(dropout))

    # keep padding to still have two distinct vectors
    model.add(Conv2D(128,(2,2),padding='same'))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(Flatten())

    # fully connected layer
    model.add(Dense(1024))
    model.add(Activation(act))
    
    # only 1 output since we are looking at real of fake, hence use sigmoid activation (softmax if using more)
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def generator_containing_discriminator(generator, discriminator):
    """
    This is the link between generator and discriminator that trains the generator to
    understand if its output is consistent with the training seen by the discriminator
    """
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

def clean_signals(s_type):
    '''
    This generates Sine Gaussian, ringdown and white noise burst signals depending on what you want.
    Also outputs the same signal but with a time delay corresponding to the light travel time between
    two detectors.
    '''
    #offset = 0.008#timeDelay(gps_time,'H1','L1') 

    if s_type == 'SG':
        A = np.random.uniform(0.2,1)                                                     # Amplitude
        f_0 = np.random.uniform(30,50)                                                   # frequency 
        t_0 =  np.random.uniform(0.25,0.75)                                              # starting epoch
        tau = np.random.uniform(1.0/60.0,1.0/15.0)                                       # decay constant 
        h = A * np.exp(-1.0*(t-t_0)**2/(tau**2))*np.sin(2*np.pi*f_0*(t-t_0) + phi)       # signal
        offset = timeDelay(gps_time,'H1','L1')                                           # time offset between two detectors 
        t_0 = t_0 + offset                                                               # update on epoch to run through same sine gaussian function
        h_offset = A * np.exp(-1.0*((t-t_0)/(tau))**2)*np.sin(2*np.pi*f_0*(t-t_0) + phi) # offset signal
    
    if s_type == 'RD':
        A = np.random.uniform(0.2,1)
        f_0 = np.random.uniform(30,50)
        t_0 = np.random.uniform(0.1,0.9)
        tau = np.random.uniform(0.02,0.1)
        offset = timeDelay(gps_time,'H1','L1')
        h = A * np.exp(-1.0*((t-t_0)/(tau)))*np.sin(2*np.pi*f_0*(t-t_0) + phi)
        h = ((t-t_0)>0)*h        							 # shifts signal by  making the start time non zero
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
        noise = np.random.normal(0,np.random.uniform(0.01,0.2),(170))
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
    return np.array(signals).reshape(batch_size,1,sample_rate,n_colors),np.array(signals_offset).reshape(batch_size,1,sample_rate,n_colors)
    

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

def combine_images(signals, nrows=5, ncols=5):
    '''
    Makes a 5x5 tile figure from a sample of a given signal array.
    '''
    fig, axs = plt.subplots(nrows, ncols, figsize=(20,10))
    #fig.suptitle("Title centered above all subplots", fontsize=14)
    fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical',fontsize=20)
    axs = axs.ravel()

    for i in range(nrows*ncols):
        axs[i].plot(t,signals[i][0],'b')						# plots the signals
        axs[i].plot(t,signals[i][1],'g')						# plots the offset
        axs[i].set_ylim([-1,1])
    return fig

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
    ax2 = fig.add_subplot(212)
    ax2.plot(logit(losses[:,1]),'g')
    if losses.shape[1]>3:
        ax2.plot(logit(losses[:,3]),'r')
    ax2.set_yticks(logit([0.001,0.01,0.1,0.5,0.9,0.99,0.999]))
    ax2.set_yticklabels(['0.001','0.01','0.1','0.5','0.9','0.99','0.999'])
    ax2.set_xlabel(r'iterations')
    ax2.set_ylabel(r'accuracy')
    if logscale==True:
        ax1.set_xscale("symlog", nonposx='clip')
        #ax1.set_xlim(0.001,10)
        ax2.set_xscale("symlog", nonposx='clip')
        ax1.set_yscale("symlog", nonposx='clip')
    plt.savefig(filename)
    plt.close('all')


def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def main():

    discriminator = discriminator_model()
    generator = generator_model()

    discriminator_on_generator = generator_containing_discriminator(generator,
 discriminator)
    set_trainable(discriminator, False) # not trainable in this step
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    print(generator.summary())
    print(discriminator_on_generator.summary())

    # setup training on discriminator 
    # using binary cross entropy for loss since we only want discrimination between real and fake images
    set_trainable(discriminator, True)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    print(discriminator.summary())
    

    losses = []
    for i in range(max_itr):
        
        batch_signal,batch_signal_delay = sig_gen(batch_size,s_type) 				# arrays of signals and their offsets
        dual_signals = np.concatenate((batch_signal,batch_signal_delay),axis=1)			# combination of both
        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)			# 100 random numbers between -1 and 1
        generated_signals = generator.predict(noise)						# makes predictions on the noise
        X = np.concatenate((dual_signals, generated_signals))					
        y = [1] * batch_size + [0] * batch_size							# labes rela images as 1 and fakes as 0

        d_loss = discriminator.train_on_batch(X, y)						# discriminator loss

        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0)			
        g_loss = discriminator_on_generator.train_on_batch(noise, [1] * batch_size)		# generator loss

        losses.append([g_loss[0], g_loss[1], d_loss[0],d_loss[1]])
       
       

        if i % 100 == 0:
           print("step%d d_loss, g_loss : %g %g" % (i, d_loss[0], g_loss[0]))
           print("step%d d_acc, g_acc : %g %g" % (i, d_loss[1], g_loss[1]))
           gen_image = combine_images(generated_signals) 
           batch_image = combine_images(dual_signals)
           gen_image.savefig(outpath % i)
           batch_image.savefig(batch_outpath % i)
           plot_losses(losses,'%s/losses_var.png' % loss_path,logscale=False, legend=['Generator','Discriminator'])
           plot_losses(losses,'%s/losses_logscale_var.png' % loss_path,logscale=True, legend=['Generator','Discriminator'])
main()


