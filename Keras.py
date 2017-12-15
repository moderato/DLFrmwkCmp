import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import model_selection as ms
from sys import platform
import DLHelper
from timeit import default_timer
from keras_resnet import ResnetBuilder

if platform == "darwin":
    root = "/Users/moderato/Downloads/GTSRB/try"
else:
    root = "/home/zhongyilin/Desktop/GTSRB/try"
print(root)
resize_size = (48, 48)
trainImages, trainLabels, testImages, testLabels = DLHelper.getImageSets(root, resize_size)
x_train, x_valid, y_train, y_valid = ms.train_test_split(trainImages, trainLabels, test_size=0.2, random_state=542)

epoch_num = 1
batch_size = 64

from keras.layers import Conv2D as keras_Conv
from keras.layers import (
    MaxPooling2D as keras_MaxPooling, 
    GlobalAveragePooling2D as keras_AveragePooling
)
from keras.layers import (
    Dropout as keras_Dropout, 
    Dense, 
    Flatten
)
from keras.models import Sequential
from keras.utils import np_utils, to_categorical
from keras import backend as K
from keras.preprocessing import image
from keras.initializers import RandomNormal, Constant as keras_Constant
from keras.optimizers import SGD as keras_SGD, RMSprop as keras_RMSProp
from keras.callbacks import ModelCheckpoint, Callback as keras_callback
from sklearn import model_selection as ms
from keras.layers.convolutional import ZeroPadding2D
from keras.models import model_from_json
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
        self.f = DLHelper.init_h5py(filename, epoch_num, max_total_batch)
    
    def on_train_begin(self, logs={}):
        try:
            self.f['.']['time']['train']['start_time'][0] = default_timer()
        except Exception as e:
            self.f.close()
            raise e

    def on_epoch_end(self, epoch, logs={}):
        try:
            print(logs)
            self.f['.']['cost']['loss'][epoch] = np.float32(logs.get('val_loss'))
            self.f['.']['accuracy']['valid'][epoch] = np.float32(logs.get('val_acc') * 100.0)
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
            self.f['.']['accuracy']['train'][self.batch_count-1] = np.float32(logs.get('acc') * 100.0)
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
        except Exception as e:
            self.f.close()
            raise e

def constructCNN(layer_name_prefix, cnn_type="self"):
    keras_model = Sequential()
    if cnn_type == "idsia":
        keras_model.add(keras_Conv(100, (3, 3), strides=(1, 1), activation="relu", input_shape=(resize_size[0], resize_size[1], 3), name=layer_name_prefix+"conv1"))
        keras_model.add(keras_MaxPooling(pool_size=(2, 2), name=layer_name_prefix+"pool1"))
        keras_model.add(keras_Conv(150, (4, 4), strides=(1, 1), activation="relu", name=layer_name_prefix+"conv2"))
        keras_model.add(keras_MaxPooling(pool_size=(2, 2), name=layer_name_prefix+"pool2"))
        keras_model.add(keras_Conv(250, (3, 3), strides=(1, 1), activation="relu", input_shape=(resize_size[0], resize_size[1], 3), name=layer_name_prefix+"conv3"))
        keras_model.add(keras_MaxPooling(pool_size=(2, 2), name=layer_name_prefix+"pool3"))
        keras_model.add(Flatten(name=layer_name_prefix+"flatten")) # An extra layer to flatten the previous layer in order to connect to fully connected layer
        keras_model.add(Dense(200, activation="relu", name=layer_name_prefix+"fc1"))
        keras_model.add(Dense(43, activation="softmax", name=layer_name_prefix+"fc2"))
    elif cnn_type == "self":
        keras_model.add(keras_Conv(64, (5, 5), strides=(2, 2), padding="same", activation="relu", input_shape=(resize_size[0], resize_size[1], 3), name=layer_name_prefix+"conv1"))
        keras_model.add(keras_MaxPooling(pool_size=(2, 2), name=layer_name_prefix+"pool1"))
        keras_model.add(keras_Conv(256, (3, 3), strides=(1, 1), padding="same", activation="relu", name=layer_name_prefix+"conv2"))
        keras_model.add(keras_MaxPooling(pool_size=(2, 2), name=layer_name_prefix+"pool2"))
        # keras_model.add(keras_AveragePooling(name=layer_name_prefix+"global_pool"))
        keras_model.add(Flatten(name=layer_name_prefix+"flatten")) # An extra layer to flatten the previous layer in order to connect to fully connected layer
        keras_model.add(Dense(2048, activation="relu", name=layer_name_prefix+"fc1"))
        keras_model.add(keras_Dropout(0.5, name=layer_name_prefix+"drop_out"))
        keras_model.add(Dense(43, activation="softmax", name=layer_name_prefix+"fc2"))
    elif cnn_type =="resnet-50":
        keras_model = ResnetBuilder.build_resnet_50((3, resize_size[0], resize_size[1]), 43)
    elif cnn_type =="resnet-34":
        keras_model = ResnetBuilder.build_resnet_34((3, resize_size[0], resize_size[1]), 43)

    return keras_model

# Function to dynamically change keras backend
from importlib import reload
def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

from sys import platform
backends = ["tensorflow"]
# if platform != "darwin":
#     backends.append("cntk")

for b in backends:
    set_keras_backend(b)

    max_total_batch = (len(x_train) / batch_size + 1) * epoch_num

    # Load and process images
    keras_train_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in x_train])
    keras_valid_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in x_valid])
    keras_test_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in testImages])
    keras_train_y = to_categorical(y_train, 43)
    keras_valid_y = to_categorical(y_valid, 43)
    keras_test_y = to_categorical(testLabels, 43)

    # Build model
    layer_name_prefix = b+"_"

    keras_model = constructCNN(layer_name_prefix, "idsia")

    keras_model.summary()

    keras_optimizer = keras_SGD(lr=0.01, decay=1.6e-8, momentum=0.9) # Equivalent to decay rate 0.2 per epoch? Need to re-verify
#     keras_optimizer = keras_RMSProp(lr=0.01, decay=0.95)
    keras_cost = "categorical_crossentropy"
    keras_model.compile(loss=keras_cost, optimizer=keras_optimizer, metrics=["acc"])

    checkpointer = ModelCheckpoint(filepath=root+"/saved_models/keras_"+b+"_weights.hdf5",
                                       verbose=1, save_best_only=True)
    losses = LossHistory(root+"/saved_data/callback_data_keras_{}.h5".format(b), epoch_num, max_total_batch)

    start = time.time()
    keras_model.fit(keras_train_x, keras_train_y,
                  validation_data=(keras_valid_x, keras_valid_y),
                  epochs=epoch_num, batch_size=batch_size, callbacks=[checkpointer, losses], verbose=1, shuffle=True)
    print("{} training finishes in {:.2f} seconds.".format(b, time.time() - start))

    keras_model.load_weights(root+"/saved_models/keras_"+b+"_weights.hdf5")
    keras_predictions = [np.argmax(keras_model.predict(np.expand_dims(feature, axis=0))) for feature in keras_test_x]

    # report test accuracy
    keras_test_accuracy = 100*np.sum(np.array(keras_predictions)==np.argmax(keras_test_y, axis=1))/len(keras_predictions)
    losses.f['.']['infer_acc']['accuracy'][0] = np.float32(keras_test_accuracy)
    losses.f.close()
    print('{} test accuracy: {:.1f}%'.format(b, keras_test_accuracy))

    json_string = keras_model.to_json()
    js = open("root+/saved_models/keras_"+b+"_config.json", "w")
    js.write(json_string)
    js.close()
