'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

# Reference

- Auto-Encoding Variational Bayes
  https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from skimage import transform as tf
import cv2,re,math
import argparse, os
from time import clock
from skimage.measure import compare_ssim

# input image dimensions
#img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
filters = 8
# convolution kernel size
num_conv = 4

batch_size = 1


latent_dim = 16
intermediate_dim = 64

# remove files from directory with wildcards
def _purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

#---> CREATE MAIN FUNCTION TO READ PAIRED IMAGES FROM COMMAND LINE
#---function to input and scale paired images
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
    
    hid_decoded = decoder_hid(z_mean)
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
    
    
    #x_train = x_train.astype('float32') / 255.
    #x_test = x_test.astype('float32') / 255.
    
    print('x_train.shape:', x_train.shape)
    
    
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
    return generator,encoder

def parse_cmd_line():
    p = argparse.ArgumentParser(description='change detector')
    p.set_defaults(verbose=0, quiet=False)
    p.add_argument('-d', '--dir', help='image directory', default='../small_pairs_tiles/')
    p.add_argument('-f', '--filename', help='filename string of image pair', default='c')
    p.add_argument('-e', '--epochs', help='number of epochs', default=400)
    args = p.parse_args()
    return args


if __name__ == '__main__':
    etime = -clock()
    #_purge('output/', 'png')   # purge output dir of old results
    args = parse_cmd_line()
    
    # input satellite data for change detection
    x_1, x_2, img_rows, img_cols, img_chns, image_pair = InputPairs(args)
    
    # normalize images x_1 and x_2 to have intensities [0,1]
    x_1 = cv2.normalize(x_1, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    x_2 = cv2.normalize(x_2, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
       
    generator1, encoder1 = mainprocess_model(args,x_1,x_2)  # train on x_1
    generator2, encoder2 = mainprocess_model(args,x_2,x_1)  # train on x_2
    
    '''
    # scikit-image and opencv2 packages.
    The score  represents the structural similarity index between the two input images. 
    This value can fall into the range [-1, 1] with a value of one being a 
    “perfect match”. The diff  image contains the actual image differences 
    between the two input images that we wish to visualize. The difference 
    image is currently represented as a floating point data type in the range 
    [0, 1] so we first convert the array to 8-bit unsigned integers in the range 
    [0, 255] (Line 26) before we can further process it using OpenCV.
    '''
    
    # calculate change map from difference of latent layers
    x_train = x_1.reshape(1, img_rows,img_cols,img_chns)
    x_test = x_2.reshape(1, img_rows,img_cols,img_chns)
    encoded_1 = encoder1.predict(x_train)
    encoded_2 = encoder2.predict(x_test)
    cMap = cv2.absdiff(encoded_1, encoded_2)   # latent space difference
    
    # straight subtraction of x_1 and x_2, absolute valued
    diffI = cv2.absdiff(x_1, x_2)  
    
    # predict on the difference latent space
    cMap_img1 = generator1.predict(cMap, batch_size=batch_size)
    cMap_img2 = generator2.predict(cMap, batch_size=batch_size)
    
    #cMap_img = cv2.absdiff(cMap_img1[0], cMap_img2[0]) 
    #cMap_img = cv2.add(cMap_img1[0], cMap_img2[0]) 
    
    image_pair.sort()
    #plt.figure(figsize=(15, 15))
    plt.imshow(x_1)
    plt.axis('off')
    plt.show()
    
    #plt.figure(figsize=(15, 15))
    plt.imshow(x_2)
    plt.axis('off')
    plt.show()
    
    # Pseudocolor is only relevant to single-channel, grayscale, luminosity 
    # images. We currently have an RGB image. Since R, G, and B are all similar 
    #plt.figure(figsize=(15, 15))
    plt.imshow(diffI[:,:,1])    # straight difference image
    plt.colorbar()
    plt.title('difference image abs(x_1 -x_2)')
    plt.axis('off')
    plt.savefig('../output/{}_diff.png'.format(image_pair[0][0:1]))
    plt.show()
    
    # threshold image to desired output
    img = cv2.absdiff(cMap_img2[0][:,:,1],cMap_img1[0][:,:,1])
    ret,th1 = cv2.threshold(img,0.1,1,cv2.THRESH_TOZERO)
    plt.imshow(th1,cmap="hot")       # latent layers diff
    plt.colorbar()
    plt.title('change map <- abs(cMap_img1[0] - cMap_img2[0])')
    plt.axis('off')
    plt.savefig('../output/cMap_img_{}_thresh.png'.format(image_pair[0][0:1]))
    plt.show()
    #print(x_1.shape)
    # threshold image to desired output
    img = cMap_img2[0]
    ret,th1 = cv2.threshold(img,0.1,1,cv2.THRESH_TOZERO)
    plt.imshow(th1,cmap="hot")       # latent layers diff
    plt.colorbar()
    plt.title('change map <- abs(encoded_1 - encoded_2)')
    plt.axis('off')
    plt.savefig('../output/cMap_enc_{}_thresh.png'.format(image_pair[0][0:1]))
    plt.show()
    
    (score, diff) = compare_ssim(cMap_img2[0], cMap_img1[0], full=True,multichannel=True)
    ret,th1 = cv2.threshold(diff[:,:,1],0.1,1,cv2.THRESH_TOZERO)
    plt.imshow(th1,cmap="hot")
    plt.colorbar()
    plt.title('change map - structural image')
    plt.axis('off')
    plt.savefig('../output/cMap_struct_{}_thresh.png'.format(image_pair[0][0:1]))
    plt.show()
    
  