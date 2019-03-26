'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

# Reference

- Auto-Encoding Variational Bayes
  https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from skimage import transform as tf
import os, cv2, re, math, random
import tensorflow as tflow
from PIL import Image
import argparse, os
from time import clock
from skimage.measure import compare_ssim
import image_slicer
from sklearn.preprocessing import StandardScaler

# input image dimensions
#img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
filters = 8    # 8
# convolution kernel size
num_conv = 2

batch_size = 1


latent_dim = 2
intermediate_dim = 16

# remove files from directory with wildcards
def _purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

#---> CREATE MAIN FUNCTION TO READ PAIRED IMAGES FROM COMMAND LINE
#---function to input and scale paired images
def InputPairs(args):
    l =  os.listdir(args.dir)
    image_pair = [x for x in l if args.filename in x]
    x_1 = cv2.imread(args.dir + image_pair[0])
    x_2 = cv2.imread(args.dir + image_pair[1])
    # need to reshape the pairs to have same dimensions -the min (row, col) pairs
    # and convert to nearest powers of two for consistent input/output dimensions
    height = min(x_1.shape[0], x_2.shape[0])
    height = int(math.pow(2,int(math.log(height, 2))))

    width = min(x_1.shape[1], x_2.shape[1]) 
    width = int(math.pow(2,int(math.log(width, 2))))

    chan = x_1.shape[2]   # no f channels, 3 for RGB
    
    x_1 = tf.resize(x_1, (height, width, chan), order=0)  # order=0, Nearest-neighbor interpolation
    x_2 = tf.resize(x_2, (height, width, chan), order=0)
  
    return x_1, x_2, height, width, chan,image_pair

# function to normalize image so that pixels lie in the range [0,1]
# for plotting the resulting image
def normalize_image(image):
    #image = image.astype('float32') / 255.
    image -= image.min() # ensure the minimal value is 0.0
    image /= image.max() # maximum value in image is now 1.0
    return image

def mainprocess_model(args,x_train,x_test):
    '''
    train and generate change map
    '''
    
    # fine-tune the model
    x_train = x_train.reshape(1, img_rows,img_cols,img_chns)
    x_test = x_test.reshape(1, img_rows,img_cols,img_chns)
    
    if K.image_data_format() == 'channels_first':
        original_img_size = (img_chns, img_rows, img_cols)   # Theano
    else: 
        original_img_size = (img_rows, img_cols, img_chns)   # TensorFlow
    
    # Plot the histogram.
    def plot_distribution(flat_diff):
        mu, std = norm.fit(flat_diff)
        plt.hist(flat_diff, bins=25, normed=True, alpha=0.6, color='g')
        
        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)
        
        plt.show()
        return mu, std, p
    
    diff = normalize_image(x_train-x_test)
    flat_diff = diff[0].flatten()
    mu, std, p = plot_distribution(flat_diff)   # Fit a normal distribution to the data:
            
    #-----------------> ENCODER MODULE <--------
    x = Input(shape=original_img_size)
    conv_1 = Conv2D(img_chns,
                    kernel_size=num_conv,
                    padding='same', activation='tanh')(x)
    conv_2 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='tanh',
                    strides=(2, 2))(conv_1)
    conv_3 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='tanh',
                    strides=1)(conv_2)
    conv_4 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='tanh',
                    strides=1)(conv_3)
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='tanh')(flat)
    
    z_mean = Dense(latent_dim)(hidden)      # generate latent_dim # of means and stds
    z_log_var = Dense(latent_dim)(hidden)
    
    #---------------------------------------------------------
       
    def sampling(args):
        z_mean, z_log_var = args
        # we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=1.0)
        # sampling from a norm fit of our diff data
        #epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
        #                         mean=mu, stddev=std)
        # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~(0,1)
        return z_mean  + K.exp(z_log_var) * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling) ([z_mean, z_log_var])    # SAMPLE FROM NORMAL DISTRIBUTION
                                                  # COULD USE THIS LAYER FOR MY FRACTIONAL FFT IDEA.
    
    # -----  > DECODER MODULE <----------------------------------------------
    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='tanh')         # LAYER BEFORE NORMAL DISTRIBUTION
    decoder_upsample = Dense(filters * int(img_rows/2) * int(img_cols/2), activation='tanh') # DECODER OUTPUT
    
    output_shape = (batch_size, int(img_rows/2), int(img_cols/2), filters)
    
    decoder_reshape = Reshape(output_shape[1:])
    decoder_deconv_1 = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
    
                                       activation='tanh')
    
    decoder_deconv_2 = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='tanh')
    
    output_shape = (batch_size, 29, 29, filters)
    decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                              kernel_size=(3, 3),
                                              strides=(2, 2),
                                              padding='valid',
                                              activation='tanh')
    decoder_mean_squash = Conv2D(img_chns,
                                 kernel_size=(2,2),
                                 padding='valid',
                                 activation='sigmoid')
    
    hid_decoded = decoder_hid(z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)
    # ------------------------------------------------------------>
    # instantiate VAE model
    vae = Model(x, x_decoded_mean_squash)
    
    # Compute VAE loss
    #xent_loss = img_rows * img_cols * metrics.binary_crossentropy(K.flatten(x),
    #    K.flatten(x_decoded_mean_squash))
    
    def mean_squared_logarithmic_error(y_true, y_pred):
        first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
        second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
        return K.mean(K.square(first_log - second_log), axis=-1)
    
    xent_loss = mean_squared_logarithmic_error(K.flatten(x), K.flatten(x_decoded_mean_squash))
    
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    vae.compile(optimizer='adam',loss='mean_squared_logarithmic_error')
    vae.summary()
    
    vae.fit(x_train,x_train,
            shuffle=True,
            epochs=args.epochs,
            batch_size=1)
    
    
    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    
    
    '''
    # Build a Digit Generator that can Sample from the Learned Distribution
    '''
    decoder_input = Input(shape=(latent_dim,))
    _hid_decoded = decoder_hid(decoder_input)
    _up_decoded = decoder_upsample(_hid_decoded)
    _reshape_decoded = decoder_reshape(_up_decoded)
    _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
    _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
    _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
    _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
    generator = Model(decoder_input, _x_decoded_mean_squash)
    return p, generator,encoder

def parse_cmd_line():
    p = argparse.ArgumentParser(description='change detector')
    p.set_defaults(verbose=0, quiet=False)
    p.add_argument('-d', '--dir', help='image directory', default='../small_pairs_tiles/')
    p.add_argument('-f', '--filename', help='filename string of image pair', default='c')
    p.add_argument('-e', '--epochs', help='number of epochs', default=200)
    args = p.parse_args()
    return args


if __name__ == '__main__':
    etime = -clock()
    #_purge('output/', 'png')   # purge output dir of old results
    args = parse_cmd_line()
    
    # input satellite data for change detection
    x_1, x_2, img_rows, img_cols, img_chns, image_pair = InputPairs(args)
    
    '''
    Standardizing the features
    Use StandardScaler to help you standardize the dataset’s 
    features onto unit scale (mean = 0 and variance = 1) which is a requirement 
    for the optimal performance of many machine learning algorithms.
    '''
    for i in range(3):
        x_1[:,:,i] = StandardScaler().fit_transform(x_1[:,:,i])
        x_2[:,:,i] = StandardScaler().fit_transform(x_2[:,:,i])
    
    
    # normalize images x_1 and x_2 to have intensities [0,1]
    x_1 = cv2.normalize(x_1,  x_1, 0, 1, cv2.NORM_MINMAX)
    x_2 = cv2.normalize(x_2,  x_2, 0, 1, cv2.NORM_MINMAX)
    
    p1, generator1, encoder1 = mainprocess_model(args,x_1,x_2)  # train on x_1
    p2, generator2, encoder2 = mainprocess_model(args,x_2,x_1)  # train on x_2
    
    
    #z_sample = np.array([[p1[0],p1[0]]])
    z_sample = np.array([p1[0:latent_dim]])  # # of samples for prediction: latent_dim
    z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
    x_decoded1 = generator1.predict(z_sample, batch_size=batch_size)
    
    #z_sample = np.array([[p2[0],p2[0]]])
    z_sample = np.array([p2[0:latent_dim]])
    x_decoded2 = generator2.predict(z_sample, batch_size=batch_size)
    
    # calculate change map from difference of latent layers
    x_train = x_1.reshape(1, img_rows,img_cols,img_chns)
    x_test = x_2.reshape(1, img_rows,img_cols,img_chns)
    encoded_1 = encoder1.predict(x_train)
    encoded_2 = encoder2.predict(x_test)
        
    cMap_img1 = generator1.predict(encoded_1, batch_size=batch_size)
    cMap_img2 = generator2.predict(encoded_2, batch_size=batch_size)
    
    encoded_diff = cv2.absdiff(encoded_1,encoded_2)
    cMap = generator1.predict(encoded_diff, batch_size=batch_size)
    
    #cMap = normalize_image((x_decoded2[0] - x_decoded1[0]))   # HOW TO DEAL WITH NEGATIVE NUMBERS
    cMap_img = cv2.absdiff(cMap_img1[0], cMap_img2[0])
    diffI = cv2.absdiff(x_1,x_2)
    
    image_pair.sort()
    plt.imshow(x_1)
    plt.axis('off')
    plt.show()
    
    plt.imshow(x_2)
    plt.axis('off')
    plt.show()
    
    # Pseudocolor is only relevant to single-channel, grayscale, luminosity 
    # images. We currently have an RGB image. Since R, G, and B are all similar 
    plt.imshow(diffI[:,:,1])
    plt.colorbar()
    plt.axis('off')
    plt.title('difference image |(x1-x2)|')
    plt.savefig('../output/{}_diff.png'.format(image_pair[0][0:1]))
    plt.show()
    
    img = cMap[0][:,:,1]
    ret,img = cv2.threshold(img,0.1,1,cv2.THRESH_TOZERO)
    plt.imshow(img, cmap="hot")
    plt.colorbar()
    plt.title('change map <- |(encoded_1 - encoded_2)|')
    plt.axis('off')
    plt.savefig('../output/dec_diff_{}.png'.format(image_pair[0][0:1]))
    plt.show()
    
    img = cMap_img[:,:,1]
    ret,th1 = cv2.threshold(img,0.1,1,cv2.THRESH_TOZERO)
    plt.imshow(th1,cmap="hot")       # latent layers diff
    plt.colorbar()
    plt.title('change map <- (cMap_img1[0] - cMap_img2[0])')
    plt.axis('off')
    plt.savefig('../output/cMap_img_{}.png'.format(image_pair[0][0:1]))
    plt.show()
    
  