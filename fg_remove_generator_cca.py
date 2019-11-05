# code for a UNet architecture for learning how to remove 
# the foreground from a 21ccm signal

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.python.client import device_lib
import pickle


from my_classes import EpochDataGenerator

def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type    == "GPU"]


def build_model():
	## Start with inputs
	inputs = keras.layers.Input(shape=(64,64,30),name="image_input")
	conv1 = keras.layers.Conv2D(64,8,activation=tf.nn.relu,name="conv1",padding="same",strides=2)(inputs)
	conv2 = keras.layers.Conv2D(128,4,activation=tf.nn.relu,name="conv2",padding="same",strides=2)(conv1)
	conv3 = keras.layers.Conv2D(256,2,activation=tf.nn.relu,name="conv3",padding="same",strides=2)(conv2)
	# symmetric upsampling path
	upconv1 = keras.layers.Conv2DTranspose(128,2,activation=tf.nn.relu,padding="same",name="upconv1",strides=2)(conv3)
	concat1 = keras.layers.concatenate([conv2,upconv1],name="concat1")
	upconv2 = keras.layers.Conv2DTranspose(64,4,padding="same",name="upconv2",strides=2)(concat1)
	concat2 =  keras.layers.concatenate([conv1,upconv2],name="concat2")
	upconv3 = keras.layers.Conv2DTranspose(64,8,padding="same",name="upconv3",strides=2)(concat2)
	concat3 =  keras.layers.concatenate([inputs,upconv3],name="concat3")
	output = keras.layers.Conv2DTranspose(30,1,padding="same",name="output")(concat3)
	model = keras.models.Model(inputs=inputs,outputs=output)
	return model

if __name__ == '__main__':
	# build model
	model = build_model()

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
	# give a summary of the model
	#print(model.summary())

	# load the training data
	data = np.load("/mnt/home/tmakinen/ceph/ska_sims/pca_reduced_nsim100.npy")
	print('data size: ', data.shape)
	signal = np.load("/mnt/home/tmakinen/ceph/ska_sims/cosmo_nsim100.npy")

	# split the data in to training/validation/testing sets
	x_train = data[:17000]
	y_train = signal[:17000]
	x_val = data[17000:-200]
	y_val = signal[17000:-200]
	x_test = data[-200:]
	y_test = signal[-200:]


	# train the model
	N_EPOCHS = 500
	N_BATCH = 100
	N_SIMS_PER_EPOCH = 1000   # number of examples to randomly sample from the full data
	t1 = time.time()

	# run data generators
	train_generator = EpochDataGenerator(x_train, y_train, batch_size=N_BATCH, num_sims=N_SIMS_PER_EPOCH)
	val_generator = EpochDataGenerator(x_val, y_val, batch_size=N_BATCH, num_sims=N_SIMS_PER_EPOCH)

	history = model.fit_generator(generator=train_generator, epochs=N_EPOCHS, validation_data=val_generator, use_multiprocessing=True,
                    workers=6)	
	t2 = time.time()

	print('total training time for ', N_EPOCHS, ' epochs : ', t2-t1)

	# save the results of the training
	model.save("./model_gen_nepoch500")

	# pickle the training history object
	outfile = './trainHistoryDictGen_nepoch500'	
	with open(outfile, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)

	file_pi.close()
