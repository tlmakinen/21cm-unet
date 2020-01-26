# code for a UNet architecture for learning how to remove 
# the foreground from a 21ccm signal

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.python.client import device_lib
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump, load


from my_classes import EpochDataGenerator

def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type    == "GPU"]

## Build the Model:
def build_model(dropout=0.0):
    ## Start with inputs
    inputs = keras.layers.Input(shape=(64,64,30),name="image_input")
    ## First Convolutional layer made up of two convolutions, the second one down-samples
    conv1a = keras.layers.Conv2D(64,16,activation=tf.nn.relu,name="conv1a",padding="same")(inputs)
    conv1a = keras.layers.BatchNormalization(axis=1)(conv1a)
    conv1b = keras.layers.Conv2D(64,8,activation=tf.nn.relu,name="conv1b",padding="same",strides=2)(conv1a)
    conv1b = keras.layers.BatchNormalization(axis=1)(conv1b)
    ## Second convolutional layer, essentially identical, increases the number of channels
    conv2a = keras.layers.Conv2D(128,8,activation=tf.nn.relu,name="conv2a",padding="same")(conv1b)
    #keras.layers.BatchNormalization(axis=1)(conv1b)
    conv2b = keras.layers.Conv2D(128,4,activation=tf.nn.relu,name="conv2b",padding="same",strides=2)(conv2a)
    conv2b = keras.layers.Dropout(dropout)(conv2b)
    conv2b = keras.layers.BatchNormalization(axis=1)(conv2b)
    ## Third, continuing logically from above
    conv3a = keras.layers.Conv2D(256,4,activation=tf.nn.relu,name="conv3a",padding="same")(conv2b)
    conv3a = keras.layers.BatchNormalization(axis=1)(conv3a)
    conv3b = keras.layers.Conv2D(256,4,activation=tf.nn.relu,name="conv3b",padding="same",strides=2)(conv3a)
    conv3b = keras.layers.Dropout(dropout)(conv3b)
    conv3b = keras.layers.BatchNormalization(axis=1)(conv3b)
    ## Fourth, again, continuing logically from above
    conv4a = keras.layers.Conv2D(512,4,activation=tf.nn.relu,name="conv4a",padding="same")(conv3b)
    conv4a = keras.layers.BatchNormalization(axis=1)(conv4a)
    conv4b = keras.layers.Conv2D(512,4,activation=tf.nn.relu,name="conv4b",padding="same",strides=2)(conv4a)
    conv4b = keras.layers.Dropout(dropout)(conv4b)
    conv4b = keras.layers.BatchNormalization(axis=1)(conv4b)
    ## middle
    #convm = keras.layers.Conv2D(512 * 2, 2, activation=tf.nn.relu, padding="same")(conv4b)
    #convm = keras.layers.Conv2D(512 * 2, 2, activation=tf.nn.relu, padding="same")(convm)
    ## symmetric upsampling path with concatenation from down-sampling 
    upconv1a = keras.layers.Conv2DTranspose(512,4,activation=tf.nn.relu,padding="same",name="upconv1a")(conv4b)
    upconv1a = keras.layers.BatchNormalization(axis=1)(upconv1a)
    upconv1b = keras.layers.Conv2DTranspose(512,4,activation=tf.nn.relu,padding="same",name="upconv1b",strides=2)(upconv1a)
    upconv1b = keras.layers.BatchNormalization(axis=1)(upconv1b)
    upconv1b = keras.layers.Dropout(dropout)(upconv1b) 
    ## The up-convolution is then concatenated with the output from "across the U" and passed along
    concat1 = keras.layers.concatenate([conv4a,upconv1b],name="concat1")
    concat1 = keras.layers.BatchNormalization(axis=1)(concat1)
    ## Second set of up-convolutions
    upconv2a = keras.layers.Conv2DTranspose(256,4,activation=tf.nn.relu,padding="same",name="upconv2a")(concat1)
    upconv2a = keras.layers.BatchNormalization(axis=1)(upconv2a)
    upconv2b = keras.layers.Conv2DTranspose(256,4,activation=tf.nn.relu,padding="same",name="upconv2b",strides=2)(upconv2a)
    upconv2b = keras.layers.Dropout(dropout)(upconv2b)
    concat2 = keras.layers.concatenate([conv3a,upconv2b],name="concat2") 
    concat2 = keras.layers.BatchNormalization(axis=1)(concat2)
    ## Third set
    upconv3a = keras.layers.Conv2DTranspose(128,4,activation=tf.nn.relu,padding="same",name="upconv3a")(concat2)
    upconv3a = keras.layers.BatchNormalization(axis=1)(upconv3a)
    upconv3b = keras.layers.Conv2DTranspose(128,8,activation=tf.nn.relu,padding="same",name="upconv3b",strides=2)(upconv3a)
    upconv3b = keras.layers.Dropout(dropout)(upconv3b)
    concat3 =  keras.layers.concatenate([conv2a,upconv3b],name="concat3") 
    concat3 = keras.layers.BatchNormalization(axis=1)(concat3)

    ## Fourth set, so the "U" has 4 layers 
    upconv4a = keras.layers.Conv2DTranspose(64,8,activation=tf.nn.relu,padding="same",name="upconv4a")(concat3)
    upconv4b = keras.layers.Conv2DTranspose(64,16,activation=tf.nn.relu,padding="same",name="upconv4b",strides=2)(upconv4a)
    upconv4b = keras.layers.Dropout(dropout)(upconv4b)
    concat4 =  keras.layers.concatenate([conv1a,upconv4b],name="concat4")
    #concat4 = keras.layers.BatchNormalization(axis=1)(concat4)
    ## Output is then put in to a shape to match the original data
    output = keras.layers.Conv2DTranspose(30,1,padding="same",name="output")(concat4)

    ## Compile the model
    model = keras.models.Model(inputs=inputs,outputs=output)
    return model

if __name__ == '__main__':
	# build model
	model = build_model()
	#model = keras.models.load_model('models_network1/model_full_1')

	N_GPU = 2
	N_GPU = len(get_available_gpus())
	print('num gpu:', N_GPU)
	try:
			model = keras.utils.multi_gpu_model(model, gpus=N_GPU)
			print("Training using multiple GPUs..")
	except:
			print("Training using single GPU or CPU..")
	# compile model with specified loss and optimizer
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True), 
								loss="mse",metrics=["accuracy"])
	# look into why the learning might be getting "stuck"
	#

	# load train / test data
	x_train = np.load('/mnt/home/tmakinen/ceph/ska_sims/pca3_nnu30_train.npy')
	y_train = np.load("/mnt/home/tmakinen/ceph/ska_sims/cosmo_nnu30_train.npy")

	x_val = np.load('/mnt/home/tmakinen/ceph/ska_sims/pca3_nnu30_val.npy')
	y_val = np.load("/mnt/home/tmakinen/ceph/ska_sims/cosmo_nnu30_val.npy")

	
	# load in scalers
	#pca_sc = load('./models_network1/cosmo_std_scaler.bin')
	#cosmo_sc = load('./models_network1/pca_std_scaler.bin')

	#num_pix = x_train.shape[0]
	#x_train = pca_sc.transform(x_train.reshape(-1,1)).reshape(num_pix, 64, 64, 30)
	#y_train = cosmo_sc.transform(y_train.reshape(-1,1)).reshape(num_pix, 64, 64, 30)
	# now for validation transform
	#num_pix = x_val.shape[0]
	#x_val = pca_sc.transform(x_val.reshape(-1,1)).reshape(num_pix, 64, 64, 30)
	#y_val = pca_sc.transform(y_val.reshape(-1,1)).reshape(num_pix, 64, 64, 30)
	#data = np.load("/mnt/home/tmakinen/ceph/ska_sims/pca_reduced_nsim100.npy")
	#print('data size: ', data.shape)
	#signal = np.load("/mnt/home/tmakinen/ceph/ska_sims/cosmo_nsim100.npy")

	#num_pix = signal.shape[0]

	# normalize the cosmo signal
	#scaler = StandardScaler()
	#scaler.fit(signal.reshape(-1,30))  # find standard scaling for each of the 30 freq bands
	#sig_flat_trans = scaler.transform(signal.reshape(-1,30))
	#signal = sig_flat_trans.reshape(num_pix, 64, 64, 30)

	# split the data in to training/validation/testing sets
	# one sky: 192 tiles
	#sky_size = 192
	#num_skies_train = 80
	#train_indx = sky_size * num_skies_train

	#x_train = data[:train_indx]
	#y_train = signal[:train_indx]
	#x_val = data[train_indx:-192]
	#y_val = signal[train_indx:-192]
	#x_test = data[-192:]
	#y_test = signal[-192:]

	# split the data in to training/validation/testing sets
	# one sky: 192 tiles
	#x_train = data[:17000]
	#y_train = signal[:17000]
	#x_val = data[17000:-200]
	#y_val = signal[17000:-200]
	#x_test = data[-200:]
	#y_test = signal[-200:]


	# train the model
	N_EPOCHS = 200
	N_BATCH = 192              # big batch size for multiple gpus  
	N_SIMS_PER_EPOCH = 9*192   # number of examples to randomly sample from the full data
	t1 = time.time()

	# run data generators
	#train_generator = EpochDataGenerator(x_train, y_train, batch_size=N_BATCH, num_sims=N_SIMS_PER_EPOCH)
	#val_generator = EpochDataGenerator(x_val, y_val, batch_size=N_BATCH, num_sims=N_SIMS_PER_EPOCH)
	history = model.fit(x_train,y_train,batch_size=N_BATCH,epochs=N_EPOCHS,validation_data=(x_val, y_val))
	#history = model.fit_generator(generator=train_generator, epochs=N_EPOCHS, validation_data=val_generator)	
	t2 = time.time()

	print('total training time for ', N_EPOCHS, ' epochs : ', t2-t1)

	# save the results of the training
	model.save("./models_network1/model_full_pca3")

	# pickle the training history object
	outfile = './models_network1/trainHistoryDict_full_pca3'	
	with open(outfile, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)

	file_pi.close()
