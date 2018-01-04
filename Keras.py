import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
import time, sys, DLHelper

print("**********************************")
print("Training on Keras")
print("**********************************")

if sys.platform == "darwin":
    root = "/Users/moderato/Downloads/"
else:
    root = "/home/zhongyilin/Desktop/"
print(root)

network_type = sys.argv[1]
if network_type == "idsia":
    resize_size = (48, 48)
else:
    resize_size = (int(sys.argv[2]), int(sys.argv[3]))
dataset = sys.argv[4]
epoch_num = int(sys.argv[5])
batch_size = int(sys.argv[6])
process = sys.argv[7]
printing = True if sys.argv[8] == '1' else False
backends = sys.argv[9:]
print("Training on {}".format(backends))

root, trainImages, trainLabels, testImages, testLabels, class_num = DLHelper.getImageSets(root, resize_size, dataset=dataset, process=process, printing=printing)
x_train, x_valid, y_train, y_valid = ms.train_test_split(trainImages, trainLabels, test_size=0.2, random_state=542)

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import model_from_json
import os
from timeit import default_timer
import keras_resnet

class LossHistory(Callback):
    def __init__(self, filename, epoch_num, max_total_batch):
        super(Callback, self).__init__()
        
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

def constructCNN(cnn_type="self"):
    keras_model = Sequential()
    if cnn_type == "idsia":
        keras_model.add(Conv2D(100, (3, 3), strides=(1, 1), activation="relu", input_shape=(resize_size[0], resize_size[1], 3), 
            kernel_initializer='he_normal', bias_initializer='zeros', name="keras_conv1"))
        keras_model.add(MaxPooling2D(pool_size=(2, 2), name="keras_pool1"))
        keras_model.add(Conv2D(150, (4, 4), strides=(1, 1), activation="relu", 
            kernel_initializer='he_normal', bias_initializer='zeros', name="keras_conv2"))
        keras_model.add(MaxPooling2D(pool_size=(2, 2), name="keras_pool2"))
        keras_model.add(Conv2D(250, (3, 3), strides=(1, 1), activation="relu", 
            kernel_initializer='he_normal', bias_initializer='zeros', name="keras_conv3"))
        keras_model.add(MaxPooling2D(pool_size=(2, 2), name="keras_pool3"))
        keras_model.add(Flatten(name="keras_flatten")) # An extra layer to flatten the previous layer in order to connect to fully connected layer
        keras_model.add(Dense(200, activation="relu", 
            kernel_initializer='he_normal', bias_initializer='zeros', name="keras_fc1"))
        keras_model.add(Dense(class_num, activation="softmax", 
            kernel_initializer='he_normal', bias_initializer='zeros', name="keras_fc2"))
    elif cnn_type == "self":
        keras_model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation="relu", input_shape=(resize_size[0], resize_size[1], 3), 
            kernel_initializer='he_normal', bias_initializer='zeros', name="keras_conv1"))
        keras_model.add(MaxPooling2D(pool_size=(2, 2), name="keras_pool1"))
        keras_model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation="relu", 
            kernel_initializer='he_normal', bias_initializer='zeros', name="keras_conv2"))
        keras_model.add(MaxPooling2D(pool_size=(2, 2), name="keras_pool2"))
        keras_model.add(Flatten(name="keras_flatten")) # An extra layer to flatten the previous layer in order to connect to fully connected layer
        keras_model.add(Dense(2048, activation="relu", 
            kernel_initializer='he_normal', bias_initializer='zeros', name="keras_fc1"))
        keras_model.add(Dropout(0.5, name="keras_dropout1"))
        keras_model.add(Dense(class_num, activation="softmax", 
            kernel_initializer='he_normal', bias_initializer='zeros', name="keras_fc2"))
    elif cnn_type == "resnet-56":
        keras_model = keras_resnet.resnet_v1((resize_size[0], resize_size[1], 3), 50, num_classes=class_num)
    elif cnn_type == "resnet-32":
        keras_model = keras_resnet.resnet_v1((resize_size[0], resize_size[1], 3), 32, num_classes=class_num)

    return keras_model

# Function to dynamically change keras backend
from importlib import reload
def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

# if sys.platform != "darwin":
#     backends.append("cntk")

device = None
if os.environ['CONDA_DEFAULT_ENV'] == "neon":
    device = "gpu"
else:
    device = "cpu"

for b in backends:
    set_keras_backend(b)

    max_total_batch = (len(x_train) // batch_size + 1) * epoch_num

    # Load and process images
    keras_train_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in x_train])
    keras_valid_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in x_valid])
    keras_test_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in testImages])
    keras_train_y = to_categorical(y_train, class_num)
    keras_valid_y = to_categorical(y_valid, class_num)
    keras_test_y = to_categorical(testLabels, class_num)

    # Build model
    keras_model = constructCNN(network_type)

    keras_model.summary()

    keras_optimizer = SGD(lr=0.01, decay=1.6e-8, momentum=0.9) # Equivalent to decay rate 0.2 per epoch? Need to re-verify
#     keras_optimizer = RMSProp(lr=0.01, decay=0.95)
    keras_cost = "categorical_crossentropy"
    keras_model.compile(loss=keras_cost, optimizer=keras_optimizer, metrics=["acc"])

    checkpointer = ModelCheckpoint(filepath="{}saved_models/{}/{}/keras_{}_{}_weights.hdf5".format(root, network_type, b, device, dataset),
                                       verbose=1, save_best_only=True)
    losses = LossHistory("{}saved_data/{}/{}/callback_data_keras_{}_{}.h5".format(root, network_type, b, device, dataset), epoch_num, max_total_batch)

    start = time.time()
    keras_model.fit(keras_train_x, keras_train_y,
                  validation_data=(keras_valid_x, keras_valid_y),
                  epochs=epoch_num, batch_size=batch_size, callbacks=[checkpointer, losses], verbose=1, shuffle=True)
    print("{} training finishes in {:.2f} seconds.".format(b, time.time() - start))

    keras_model.load_weights("{}saved_models/{}/{}/keras_{}_{}_weights.hdf5".format(root, network_type, b, device, dataset)) # Load the best model (not necessary the latest one)
    keras_predictions = [np.argmax(keras_model.predict(np.expand_dims(feature, axis=0))) for feature in keras_test_x]

    # report test accuracy
    keras_test_accuracy = 100 * np.sum(np.array(keras_predictions)==np.argmax(keras_test_y, axis=1))/len(keras_predictions)
    losses.f['.']['infer_acc']['accuracy'][0] = np.float32(keras_test_accuracy)
    losses.f.close()
    print('{} test accuracy: {:.1f}%'.format(b, keras_test_accuracy))

    json_string = keras_model.to_json()
    js = open("{}saved_models/{}/{}/keras_{}_{}_config.json".format(root, network_type, b, device, dataset), "w")
    js.write(json_string)
    js.close()
