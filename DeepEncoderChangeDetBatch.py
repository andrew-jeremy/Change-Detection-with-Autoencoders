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

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, BatchNormalization,Activation
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import  Conv2DTranspose
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

batch_size = 1
epochs = 3000   # 400
layers = 15    # number of layers in encoder/decoder

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
    #l.remove('.DS_Store')
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

def mainprocess_model(args,x_train):
    '''
    train and generate change map
    '''
    
    # fine-tune the model
    img_chns = 3
    x_train = x_train.reshape(2, img_rows,img_cols,img_chns)
    
    if K.image_data_format() == 'channels_first':
        original_img_size = (img_chns, img_rows, img_cols)   # Theano
    else: 
        original_img_size = (img_rows, img_cols, img_chns)   # TensorFlow
    
   
        
    #-----------------> ENCODER MODULE <--------
    input_img = Input(shape=original_img_size)
    x = Conv2D(256, (2, 2), padding='same')(input_img) # 64
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)        # 32
    x = Conv2D(128, (2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (2, 2), padding='same')(x)          # 16
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    
    x = Conv2D(64, (2, 2), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(img_chns, (2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)
    
    autoencoder = Model(input_img, decoded)   # autoencoder
    
    # Let's also create a separate encoder model:
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)       # encoder 
    
    # create the decoder model
    l = list(encoded.get_shape()) # ex [Dimension(None), Dimension(4), Dimension(4), Dimension(8)]
    encode = Input(shape=(int(str(l[1])), int(str(l[2])), int(str(l[3]))))    # latent layer shaped)
    
    #decoder - n layers of weights - set above top level
    
    lays = layers * -1
    decode = autoencoder.layers[lays](encode)
    lays +=1
    for i in range(lays,0):
        decode = autoencoder.layers[i](decode)
    # create the decoder model
    decoder = Model(encode, decode)         # decoder
    
    
    
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    #autoencoder.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    # print out autencoder parameters for debugging
    print(autoencoder.summary())
    
    autoencoder.fit(x_train, x_train,epochs = args.epochs, batch_size = batch_size, shuffle=True)
    return autoencoder, decoder,encoder

def parse_cmd_line():
    p = argparse.ArgumentParser(description='change detector')
    p.set_defaults(verbose=0, quiet=False)
    p.add_argument('-d', '--dir', help='image directory', default='../small_pairs_tiles/')
    p.add_argument('-f', '--filename', help='filename string of image pair', default='c')
    p.add_argument('-e', '--epochs', help='number of epochs', default=2000)
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
        
    # difference image
    #diffI = cv2.absdiff(x_1,x_2)
    diffI = cv2.subtract(x_1,x_2)
    diffI = normalize_image(diffI)
    
    #d3= np.array([x_1[:,:,0],x_1[:,:,1],x_1[:,:,2],x_2[:,:,0],x_2[:,:,1],x_2[:,:,2]])
    d3= np.array([x_1,x_2])
    autoencoder, generator1, encoder1 = mainprocess_model(args,d3)  # train on x_1
    #generator2, encoder2 = mainprocess_model(args,x_2,flag=0)  # train on x_2
    #generator3, encoder3,p3 = mainprocess_model(args,x_1,x_2,flag=1)
    
    #figure = np.zeros((digit_size * n, digit_size * n))
    # lsample from the original p  Gaussian distribution to which the diff image was mapped
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    # WE SAMPLE FROM THE SAME GAUSSIAN WITH MEAN AND STD AS FROM AFTER LATENT LAYER IN ENCODER
         
    # calculate change map from difference of latent layers
    img_chns = 3
    #x_train = x_1[:,:,0].reshape(1, img_rows,img_cols,img_chns)
    #x_test = x_2[:,:,0].reshape(1, img_rows,img_cols,img_chns)
    
    x_train = x_1.reshape(1, img_rows,img_cols,img_chns)
    x_test = x_2.reshape(1, img_rows,img_cols,img_chns)
    encoded_1 = encoder1.predict(x_train)
    encoded_2 = encoder1.predict(x_test)
    
    cMap_img1 = autoencoder.predict(x_train) 
    cMap_img2 = autoencoder.predict(x_test)  
    
    #cMap = normalize_image((x_decoded2[0] - x_decoded1[0]))   # HOW TO DEAL WITH NEGATIVE NUMBERS
    #cMap_img = normalize_image((cMap_img1[0] - cMap_img2[0])) 
    cMap_img = cv2.absdiff(cMap_img1[0], cMap_img2[0])
    
    #z_sample = cv2.subtract(z_sample1,z_sample2)
    encoded = cv2.absdiff(encoded_1,encoded_2)
    cMap = generator1.predict(encoded, batch_size=batch_size)
    
    image_pair.sort()
    #plt.figure(figsize=(15, 15))
    plt.imshow(x_1)
    plt.show()
    plt.axis('off')
    
    #plt.figure(figsize=(15, 15))
    plt.imshow(x_2)
    plt.show()
    plt.axis('off')
    # Pseudocolor is only relevant to single-channel, grayscale, luminosity 
    # images. We currently have an RGB image. Since R, G, and B are all similar 
    #plt.figure(figsize=(15, 15))
    diffI = cv2.absdiff(x_1,x_2)
    plt.imshow(diffI[:,:,1],cmap="hot")
    plt.colorbar()
    plt.title('difference image abs(x_1 -x_2)')
    plt.axis('off')
    plt.savefig('../output/{}_diff.png'.format(image_pair[0][0:1]))
    plt.show()
    
    #plt.figure(figsize=(15, 15))
    img = cMap[0,:,:,:]
    ret,img = cv2.threshold(img,0.08,1,cv2.THRESH_TOZERO)
    plt.imshow(img, cmap="hot")
    plt.colorbar()
    plt.title('change map <- (encoded_1 - encoded_2)')
    plt.axis('off')
    plt.savefig('../output/dec_diff_{}.png'.format(image_pair[0][0:1]))
    plt.show()
    
    #plt.figure(figsize=(15, 15))
    #img = cMap_img[0,:,:,0]
    img = cMap_img
    ret,th1 = cv2.threshold(img,0.1,1,cv2.THRESH_TOZERO)
    plt.imshow(th1,cmap="hot")       # latent layers diff
    plt.colorbar()
    plt.title('change map <- abs(cMap_img1[0] - cMap_img2[0])')
    plt.axis('off')
    plt.savefig('../output/cMap_img_{}.png'.format(image_pair[0][0:1]))
    plt.show()
    
    out = Image.fromarray((x_1*255).astype('uint8'),'RGB')
    out.save('../output/{}.png'.format(image_pair[0][0:2]))
    out = Image.fromarray((x_2*255).astype('uint8'),'RGB')
    out.save('../output/{}.png' .format(image_pair[1][0:2]))
    
    #plt.figure(figsize=(15, 15))
    (score, diff) = compare_ssim(cMap_img1[0], cMap_img2[0], full=True, multichannel=True)
    diff = (diff * 255).astype("uint8")
    plt.imshow(diff[:,:,0],cmap="hot"), plt.colorbar()
    plt.title('change map - structural image')
    plt.axis('off')
    plt.savefig('../output/cmp_ssim.png')
    plt.show() 