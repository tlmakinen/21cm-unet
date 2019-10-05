''' foreground removal unet file 
'''

## U-Net architecture in Tensorflow's Keras API
## to remove foreground from cosmological simulations
## of 21cm observations
## by Lachlan Lancaster


## Import the required Libraries
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

config = tf.ConfigProto( device_count = {'GPU': 4 , 'CPU': 112} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

## Build the Model:

## Start with inputs
inputs = keras.layers.Input(shape=(64,64,30),name="image_input")

## First Convolutional layer made up of two convolutions, the second one down-samples
conv1a = keras.layers.Conv2D(64,16,activation=tf.nn.relu,name="conv1a",padding="same")(inputs)
conv1b = keras.layers.Conv2D(64,8,activation=tf.nn.relu,name="conv1b",padding="same",strides=2)(conv1a)
## Second convolutional layer, essentially identical, increases the number of channels
conv2a = keras.layers.Conv2D(128,8,activation=tf.nn.relu,name="conv2a",padding="same")(conv1b)
conv2b = keras.layers.Conv2D(128,4,activation=tf.nn.relu,name="conv2b",padding="same",strides=2)(conv2a)
## Third, continuing logically from above
conv3a = keras.layers.Conv2D(256,4,activation=tf.nn.relu,name="conv3a",padding="same")(conv2b)
conv3b = keras.layers.Conv2D(256,4,activation=tf.nn.relu,name="conv3b",padding="same",strides=2)(conv3a)
## Fourth, again, continuing logically from above
conv4a = keras.layers.Conv2D(512,4,activation=tf.nn.relu,name="conv4a",padding="same")(conv3b)
conv4b = keras.layers.Conv2D(512,4,activation=tf.nn.relu,name="conv4b",padding="same",strides=2)(conv4a)
## symmetric upsampling path with concatenation from down-sampling 
upconv1a = keras.layers.Conv2DTranspose(512,4,activation=tf.nn.relu,padding="same",name="upconv1a")(conv4b)
upconv1b = keras.layers.Conv2DTranspose(512,4,activation=tf.nn.relu,padding="same",name="upconv1b",strides=2)(upconv1a)
## The up-convolution is then concatenated with the output from "across the U" and passed along
concat1 = keras.layers.concatenate([conv4a,upconv1b],name="concat1")
## Second set of up-convolutions
upconv2a = keras.layers.Conv2DTranspose(256,4,activation=tf.nn.relu,padding="same",name="upconv2a")(concat1)
upconv2b = keras.layers.Conv2DTranspose(256,4,activation=tf.nn.relu,padding="same",name="upconv2b",strides=2)(upconv2a)
concat2 = keras.layers.concatenate([conv3a,upconv2b],name="concat2")
## Third set
upconv3a = keras.layers.Conv2DTranspose(128,4,activation=tf.nn.relu,padding="same",name="upconv3a")(concat2)
upconv3b = keras.layers.Conv2DTranspose(128,8,activation=tf.nn.relu,padding="same",name="upconv3b",strides=2)(upconv3a)
concat3 =  keras.layers.concatenate([conv2a,upconv3b],name="concat3")
## Fourth set, so the "U" has 4 layers 
upconv4a = keras.layers.Conv2DTranspose(64,8,activation=tf.nn.relu,padding="same",name="upconv4a")(concat3)
upconv4b = keras.layers.Conv2DTranspose(64,16,activation=tf.nn.relu,padding="same",name="upconv4b",strides=2)(upconv4a)
concat4 =  keras.layers.concatenate([conv1a,upconv4b],name="concat4")

## Output is then put in to a shape to match the original data
output = keras.layers.Conv2DTranspose(30,1,padding="same",name="output")(concat4)

## Compile the model
model = keras.models.Model(inputs=inputs,outputs=output)

#model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


## Now make the model parallelize over multiple GPUs
parallel_model = keras.utils.multi_gpu_model(model, gpus=4)
## Compile parallel model
parallel_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])  # TRY: categorical_crossentropy loss


dirstr = "/tigress/tmakinen/ska_sims"
NUM_SIMS = 100
## Load up the data for training this is the "x" of the f(x) = y that we're training
data = np.load("%s/pca_reduced_nsim%d.npy"%(dirstr,NUM_SIMS))

## Load up the cosmological signal, this is the "y" in f(x) = y
#obs_true = np.load("%s/observed_nsim%d.npy"%(dirstr,NUM_SIMS))
#fgd_true = np.load("%s/fg_nsim%d.npy"%(dirstr,NUM_SIMS))

#signal = obs_true - fgd_true

signal = np.load("%s/cosmo_nsim%d.npy"%(dirstr,NUM_SIMS))


## Split your data in to training, validation, and testing sets
(N,M) = (18000,-1000)
x_train = data[:N]
y_train = signal[:N]
x_val = data[N:M]
y_val = signal[N:M]
x_test = data[M:]
y_test = signal[M:]

## Train the model
N_EPOCHS = 100
N_BATCH = 64
history = parallel_model.fit(x_train,y_train,batch_size=N_BATCH,epochs=N_EPOCHS,validation_data=(x_val, y_val))

# Save the results of the training
parallel_model.save("/home/tmakinen/repositories/models/first_model.h5")

## Evaluate the Model, if you haven't trained the model, it should do pretty poorly!! But it should still work
y_pred = parallel_model.predict(x_test)

xval_c1 = x_test.transpose()[15].transpose()
yval_c1 = y_test.transpose()[15].transpose()
y_c1_pred = y_pred.transpose()[15].transpose()

plt.rc('font', **{'size': 10, 'sans-serif': ['Helvetica'], 'family': 'sans-serif'})                                          
plt.rc("text.latex", preamble=["\\usepackage{helvet}\\usepackage[T1]{fontenc}"])#\\usepackage{sfmath}"])
plt.rc("text", usetex=True)
plt.rc('ps', usedistiller='xpdf')
plt.rc('savefig', **{'dpi': 300})

fig = plt.figure(figsize=(16,4))
pick = 0
ax1 = plt.subplot(131)
plt.imshow(xval_c1[pick])
plt.colorbar()
ax1.set_xticks([])
ax1.set_yticks([])
plt.title("PCA Reduced")
ax1 = plt.subplot(132)
plt.imshow(y_c1_pred[pick])
plt.colorbar()
ax1.set_xticks([])
ax1.set_yticks([])
plt.title("UNet Prediction")
ax1 = plt.subplot(133)
plt.imshow(yval_c1[pick])
plt.colorbar()
ax1.set_xticks([])
ax1.set_yticks([])
plt.title("True Cosmological Signal")
plt.gcf().set_size_inches((3.7* 3.37, 3.37))
plt.tight_layout()
plt.savefig("./figures/comparison_nsim100_nepochs%s.png"%(N_EPOCHS))