"""
Script for organizing Keras classes, such per-epoch x_data generators

T Lucas Makinen
"""
import numpy as np
from tensorflow import keras



class EpochDataGenerator(keras.utils.Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, x_data, y_data, batch_size=64, x_dim=(64,64,30), y_dim=(64, 64, 30), n_channels=1, num_sims=10,
                         shuffle=True):
        'Initialization'
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.num_sims = num_sims  # number of simlations chosen at random per epoch
        self.y_data = y_data  
        self.x_data = x_data
        self.n_channels = n_channels
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        #return 1 # one random 1000-sim batch per epoch
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of x_data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        x_data_temp = [self.x_data[k] for k in indexes]
        y_data_temp = [self.y_data[k] for k in indexes]

        # Generate x_data
        X, y = self.__x_data_generation(x_data_temp, y_data_temp)
        
        #print('y input shapes: ', np.array(y_data_temp).shape)
        #print(indexes)
        #print('X max : ', np.max(X))
        #print('y shape : ', y.shape)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        indexes = np.arange(len(self.x_data))
        # select num_sims subset of full dataset each epoch
        self.indexes = np.random.choice(indexes, size=self.num_sims)
        #if self.shuffle == True:
            #np.random.shuffle(self.indexes)

    def __x_data_generation(self, x_data_temp, y_data_temp):
        'Generates x_data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.x_dim))
        y = np.empty((self.batch_size, *self.y_dim))

        # Generate x_data
        for i in range(len(x_data_temp)):
            # Store x data
            X[i,] = x_data_temp[i] 
            y[i,] = y_data_temp[i]

        return X, y




class EpochDataGenNetwork2(keras.utils.Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, x_data, y_data, batch_size=64, x_dim=(64,64,30), y_dim=(64, 64, 30), n_channels=1, num_sims=10,
                         shuffle=True):
        'Initialization'
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.num_sims = num_sims  # number of simlations chosen at random per epoch
        self.fg_data = y_data[0]  
        self.cosmo_data = y_data[1]
        self.x_data = x_data
        self.n_channels = n_channels
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        #return 1 # one random 1000-sim batch per epoch
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of x_data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        x_data_temp = [self.x_data[k] for k in indexes]
        fg_data_temp = [self.fg_data[k] for k in indexes]
        cosmo_data_temp = [self.cosmo_data[k] for k in indexes]

        # Generate x_data
        X, fg_data, cosmo_data = self.__x_data_generation(x_data_temp, fg_data_temp, cosmo_data_temp)


        y = [fg_data, cosmo_data]
        
        print('x', x_data_temp)
        #print(indexes)
        #print('X max : ', np.max(X))
        #print('y shape : ', y.shape)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        indexes = np.arange(len(self.x_data))
        # select num_sims subset of full dataset each epoch
        self.indexes = np.random.choice(indexes, size=self.num_sims)
        #if self.shuffle == True:
            #np.random.shuffle(self.indexes)

    def __x_data_generation(self, x_data_temp, fg_data_temp, cosmo_data_temp):
        'Generates x_data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.x_dim))
        fg_data = np.empty((self.batch_size, *self.y_dim))
        cosmo_data = np.empty((self.batch_size, *self.y_dim))

        # Generate x_data
        for i in range(len(x_data_temp)):
            # Store x data
            X[i,] = x_data_temp[i] 
            cosmo_data[i,] = cosmo_data_temp[i]
            fg_data[i,] = fg_data_temp[i]

        return X, fg_data, cosmo_data