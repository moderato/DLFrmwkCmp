import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import model_selection as ms
from sys import platform
from prp_img import getImageSets

if platform == "darwin":
    root = "/Users/moderato/Downloads/GTSRB/try"
else:
    root = "/home/zhongyilin/Desktop/GTSRB/try"
print(root)
resize_size = (49, 49)
trainImages, trainLabels, testImages, testLabels = getImageSets(root, resize_size)
x_train, x_valid, y_train, y_valid = ms.train_test_split(trainImages, trainLabels, test_size=0.2, random_state=542)

epoch_num = 1
batch_size = 128

from keras.layers import Conv2D as keras_Conv
from keras.layers import MaxPooling2D as keras_MaxPooling, GlobalAveragePooling2D as keras_AveragePooling
from keras.layers import Dropout as keras_Dropout, Dense, Flatten
from keras.models import Sequential
from keras.utils import np_utils, to_categorical
from keras import backend as K
from keras.preprocessing import image
from keras.initializers import RandomNormal, Constant as keras_Constant
from keras.optimizers import SGD as keras_SGD, RMSprop as keras_RMSProp
from keras.callbacks import ModelCheckpoint, Callback as keras_callback
from sklearn import model_selection as ms
from sklearn.preprocessing import OneHotEncoder
from keras.layers.convolutional import ZeroPadding2D
import os, h5py
from timeit import default_timer

class LossHistory(keras_callback):
    def __init__(self, filename, epoch_num, max_total_batch):
        super(keras_callback, self).__init__()
        
        self.batch_count = 0
        self.epoch_num = epoch_num
        self.filename = filename
        self.batch_time = None
        self.max_total_batch = max_total_batch
        
        self.f = h5py.File(filename, 'w')
        
        try:
            config = self.f.create_group('config')
            config.attrs["total_epochs"] = self.epoch_num

            cost = self.f.create_group('cost')
            loss = cost.create_dataset('loss', (self.epoch_num,))
            loss.attrs['time_markers'] = 'epoch_freq'
            loss.attrs['epoch_freq'] = 1
            train = cost.create_dataset('train', (self.max_total_batch,)) # Set size to maximum theoretical value
            train.attrs['time_markers'] = 'minibatch'

            t = self.f.create_group('time')
            loss = t.create_dataset('loss', (self.epoch_num,))
            train = t.create_group('train')
            start_time = train.create_dataset("start_time", (1,))
            start_time.attrs['units'] = 'seconds'
            end_time = train.create_dataset("end_time", (1,))
            end_time.attrs['units'] = 'seconds'
            train_batch = t.create_dataset('train_batch', (self.max_total_batch,)) # Same as above

            time_markers = self.f.create_group('time_markers')
            time_markers.attrs['epochs_complete'] = self.epoch_num
            train_batch = time_markers.create_dataset('minibatch', (self.epoch_num,))
        except Exception as e:
            self.f.close() # Avoid hdf5 runtime error or os error
            raise e # Catch the exception to close the file, then raise it to stop the program
    
    def on_train_begin(self, logs={}):
        try:
            self.f['.']['time']['train']['start_time'][0] = default_timer()
        except Exception as e:
            self.f.close()
            raise e

    def on_epoch_end(self, epoch, logs={}):
        try:
            self.f['.']['cost']['loss'][epoch] = np.float32(logs.get('val_loss'))
            self.f['.']['time_markers']['minibatch'][epoch] = np.float32(self.batch_count)
        except Exception as e:
            self.f.close()
            raise e
        
    def on_batch_begin(self, batch, logs={}):
        try:
            self.batch_time = default_timer()
        except Exception as e:
            self.f.close()
            raise e
    
    def on_batch_end(self, batch, logs={}):
        try:
            self.f['.']['cost']['train'][self.batch_count] = np.float32(logs.get('loss'))
            self.f['.']['time']['train_batch'][self.batch_count] = (default_timer() - self.batch_time)
            self.batch_count += 1
        except Exception as e:
            self.f.close()
            raise e
        
    def on_train_end(self, logs=None):
        try:
            self.f['.']['time']['train']['end_time'][0] = default_timer()
            self.f['.']['config'].attrs["total_minibatches"] = self.batch_count
            self.f['.']['time_markers'].attrs['minibatches_complete'] = self.batch_count
            # self.f['.']['time']['train_batch'].resize((self.batch_count,))
            self.f.close()
        except Exception as e:
            self.f.close()
            raise e

# Function to dynamically change keras backend
from importlib import reload
def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

from sys import platform
backends = ["tensorflow", "theano"]
if platform != "darwin":
    backends.append("cntk")

for b in backends:
    set_keras_backend(b)

    max_total_batch = (len(x_train) / batch_size + 1) * epoch_num

    # Load and process images
    enc = OneHotEncoder(sparse=False)
    x_train, x_valid, y_train, y_valid = ms.train_test_split(trainImages, trainLabels, test_size=0.2, random_state=542)
    keras_train_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in x_train])
    keras_valid_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in x_valid])
    keras_test_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in testImages])
    keras_train_y = to_categorical(y_train, 43)
    keras_valid_y = to_categorical(y_valid, 43)
    keras_test_y = to_categorical(testLabels, 43)

    # Build model
    layer_name_prefix = b+"_"

    keras_model = Sequential()
    keras_model.add(keras_Conv(64, (5, 5), kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=542), strides=(2, 2), bias_initializer=keras_Constant(0.0), activation="relu", input_shape=(resize_size[0], resize_size[1], 3), name=layer_name_prefix+"conv1"))
    keras_model.add(keras_MaxPooling(pool_size=(2, 2), name=layer_name_prefix+"pool1"))
    keras_model.add(keras_Conv(256, (3, 3), kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=542), strides=(1, 1), padding="same", bias_initializer=keras_Constant(0.0), activation="relu", name=layer_name_prefix+"conv2"))
    keras_model.add(keras_MaxPooling(pool_size=(2, 2), name=layer_name_prefix+"pool2"))
    keras_model.add(keras_AveragePooling(name=layer_name_prefix+"global_pool"))
#     keras_model.add(Flatten(name=layer_name_prefix+"flatten")) # An extra layer to flatten the previous layer in order to connect to fully connected layer
#     keras_model.add(Dense(4096, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=542), bias_initializer=keras_Constant(0.0), activation="relu", name=layer_name_prefix+"fc1"))
    keras_model.add(keras_Dropout(0.5, name=layer_name_prefix+"drop_out"))
    keras_model.add(Dense(43, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=542), bias_initializer=keras_Constant(0.0), activation="softmax", name=layer_name_prefix+"fc2"))
    keras_model.summary()

    keras_optimizer = keras_SGD(lr=0.01, decay=1.6e-8, momentum=0.9) # Equivalent to decay rate 0.2 per epoch? Need to re-verify
#     keras_optimizer = keras_RMSProp(lr=0.01, decay=0.95)
    keras_cost = "categorical_crossentropy"
    keras_model.compile(loss=keras_cost, optimizer=keras_optimizer, metrics=["acc"])

    checkpointer = ModelCheckpoint(filepath=root+"/saved_models/keras_"+b+"_weights.hdf5",
                                       verbose=1, save_best_only=True)
    losses = LossHistory(root+"/callback_data_{}.h5".format(b), epoch_num, max_total_batch)

    start = time.time()
    keras_model.fit(keras_train_x, keras_train_y,
                  validation_data=(keras_valid_x, keras_valid_y),
                  epochs=epoch_num, batch_size=batch_size, callbacks=[checkpointer, losses], verbose=1, shuffle=True)
    print("{} training finishes in {:.2f} seconds.".format(b, time.time() - start))

    keras_model.load_weights(root+"/saved_models/keras_"+b+"_weights.hdf5")
    keras_predictions = [np.argmax(keras_model.predict(np.expand_dims(feature, axis=0))) for feature in keras_test_x]

    # report test accuracy
    keras_test_accuracy = 100*np.sum(np.array(keras_predictions)==np.argmax(keras_test_y, axis=1))/len(keras_predictions)
    print('{} test accuracy: {:.1f}%'.format(b, keras_test_accuracy))
