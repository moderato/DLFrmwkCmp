import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import model_selection as ms
from sys import platform
import DLHelper
from timeit import default_timer

if platform == "darwin":
    root = "/Users/moderato/Downloads/"
else:
    root = "/home/zhongyilin/Desktop/"
print(root)

resize_size = (48, 48)
dataset = "GT"

root, trainImages, trainLabels, testImages, testLabels, class_num = DLHelper.getImageSets(root, resize_size, dataset=dataset, printing=False)
x_train, x_valid, y_train, y_valid = ms.train_test_split(trainImages, trainLabels, test_size=0.2, random_state=542)

epoch_num = 2
batch_size = 64

import cntk as C
from cntk.learners import momentum_sgd as cntk_SGD
from cntk import cross_entropy_with_softmax as cntk_softmax, classification_error as cntk_error
from cntk.io import MinibatchSourceFromData
from cntk.logging import ProgressPrinter
from cntk.train.training_session import *
from cntk.initializer import xavier
from cntk_resnet import *
from timeit import default_timer
import os

backend = 'CPU'
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))
        backend = 'GPU'

def constructCNN(cntk_input, layer_name_prefix, cnn_type='self'):
    model = None
    if cnn_type == 'idsia':
        with C.layers.default_options(init=C.normal(0.01), activation=C.relu):
            model = C.layers.Sequential([
                C.layers.Convolution((3,3), strides=(1,1), num_filters=100, pad=False,
                    name=layer_name_prefix+"conv1"),
                C.layers.MaxPooling((2,2), strides=(2,2), name=layer_name_prefix+"pool1"),
                C.layers.Convolution((4,4), strides=(1,1), num_filters=150, pad=False,
                    name=layer_name_prefix+"conv2"),
                C.layers.MaxPooling((2,2), strides=(2,2), name=layer_name_prefix+"pool2"),
                C.layers.Convolution((3,3), strides=(1,1), num_filters=250, pad=False,
                    name=layer_name_prefix+"conv3"),
                C.layers.MaxPooling((2,2), strides=(2,2), name=layer_name_prefix+"pool3"),

                C.layers.Dense(200, name=layer_name_prefix+"fc1"),
                C.layers.Dense(class_num, activation=None, name=layer_name_prefix+"fc2") # Leave the softmax for now
            ])(cntk_input)
    elif cnn_type == 'self':
        with C.layers.default_options(init=C.normal(0.01), activation=C.relu):
            model = C.layers.Sequential([
                C.layers.Convolution((5,5), strides=(2,2), num_filters=64, pad=True,
                    name=layer_name_prefix+"conv1"),
                C.layers.MaxPooling((2,2), strides=(2,2), name=layer_name_prefix+"pool1"),
                C.layers.Convolution((3,3), strides=(1,1), num_filters=256, pad=True,
                    name=layer_name_prefix+"conv2"),
                C.layers.MaxPooling((2,2), strides=(2,2), name=layer_name_prefix+"pool2"),

                C.layers.Dense(2048, name=layer_name_prefix+"fc1"),
                C.layers.Dropout(0.5, name=layer_name_prefix+"dropout1"),
                C.layers.Dense(class_num, activation=None, name=layer_name_prefix+"fc2") # Leave the softmax for now
            ])(cntk_input)
    elif cnn_type == "resnet-56":
        cntk_resnet.create_model(cntk_input, 9, class_num) # 6*9 + 2 = 56
    elif cnn_type == "resnet-32":
        cntk_resnet.create_model(cntk_input, 5, class_num) # 6*5 + 2 = 32
    
    return model

# Construct model, io and metrics
cntk_input = C.input_variable((3, resize_size[0], resize_size[1]), np.float32)
cntk_output = C.input_variable((class_num), np.float32)
cntk_model = constructCNN(cntk_input, 'cntk_', 'idsia')
cntk_cost = cntk_softmax(cntk_model, cntk_output)
cntk_error = cntk_error(cntk_model, cntk_output)


# Construct data
cntk_train_x = np.vstack([np.expand_dims(x, axis=0).transpose([0,3,1,2]).astype('float32')/255 for x in x_train])
cntk_valid_x = np.vstack([np.expand_dims(x, axis=0).transpose([0,3,1,2]).astype('float32')/255 for x in x_valid])
cntk_test_x = np.vstack([np.expand_dims(x, axis=0).transpose([0,3,1,2]).astype('float32')/255 for x in testImages])

cntk_train_y = C.one_hot(C.input_variable(1), class_num, sparse_output=False)(np.expand_dims(np.array(y_train, dtype='f'), axis=1))
cntk_valid_y = C.one_hot(C.input_variable(1), class_num, sparse_output=False)(np.expand_dims(np.array(y_valid, dtype='f'), axis=1))
cntk_test_y = C.one_hot(C.input_variable(1), class_num, sparse_output=False)(np.expand_dims(np.array(testLabels, dtype='f'), axis=1))


progress_writers = [ProgressPrinter(
        tag='Training',
        num_epochs=epoch_num)]

# Trainer and mb source
cntk_learner = cntk_SGD(cntk_model.parameters, lr=0.01, momentum=0.9)
cntk_trainer = C.Trainer(cntk_model, (cntk_cost, cntk_error), cntk_learner, progress_writers)
cntk_train_src = C.io.MinibatchSourceFromData(dict(x=C.Value(cntk_train_x), y=C.Value(cntk_train_y)), max_samples=len(cntk_train_x))
cntk_valid_src = C.io.MinibatchSourceFromData(dict(x=C.Value(cntk_valid_x), y=C.Value(cntk_valid_y)),max_samples=len(cntk_valid_x))
cntk_test_src = C.io.MinibatchSourceFromData(dict(x=C.Value(cntk_test_x), y=C.Value(cntk_test_y))max_samples=len(cntk_test_x))

# Mapping for training, validation and testing
train_map = {
    cntk_input: cntk_train_src.streams['x'],
    cntk_output: cntk_train_src.streams['y']
}
valid_map = {
    cntk_input: cntk_valid_src.streams['x'],
    cntk_output: cntk_valid_src.streams['y']
}
test_map = {
    cntk_input: cntk_test_src.streams['x'],
    cntk_output: cntk_test_src.streams['y']
}

# Create log file
train_batch_count = (len(x_train) // batch_size + 1) * epoch_num
valid_batch_count = len(x_valid) // batch_size + 1
test_batch_count = len(testImages) // batch_size + 1
filename = "{}/saved_data/callback_data_pytorch_{}_{}.h5".format(root, backend, dataset)
f = DLHelper.init_h5py(filename, epoch_num, train_batch_count)

# Start training
try:
    batch_count = 0
    restart_checkpoint_train = {'cursor': 0, 'total_num_samples': len(cntk_train_x)}
    restart_checkpoint_valid = {'cursor': 0, 'total_num_samples': len(cntk_valid_x)}
    restart_checkpoint_test = {'cursor': 0, 'total_num_samples': len(cntk_test_x)}
    f['.']['time']['train']['start_time'][0] = time.time()

    # Each epoch
    for epoch in range(0, epoch_num):
        cntk_train_src.restore_from_checkpoint(restart_checkpoint_train)
        cntk_valid_src.restore_from_checkpoint(restart_checkpoint_valid)
        cntk_test_src.restore_from_checkpoint(restart_checkpoint_test)

        # Each batch
        for i in range(train_batch_count / epoch_num):
            batch_count += 1

            # Read a mini batch from the training data file
            data = cntk_train_src.next_minibatch(batch_size, input_map=train_map)

            # Train a batch
            start = default_timer()
            cntk_trainer.train_minibatch(data)
            # Save batch time
            train_batch_time = default_timer() - start
            f['.']['time']['train_batch'][batch_count-1] = train_batch_time

            # Save training loss
            training_loss = cntk_trainer.previous_minibatch_loss_average
            eval_error = cntk_trainer.previous_minibatch_evaluation_average
            f['.']['cost']['train'][batch_count-1] = np.float32(training_loss)
            f['.']['accuracy']['train'][batch_count-1] = np.float32((1.0 - eval_error) * 100.0)

            print("Epoch: {0}, Minibatch: {1}, Loss: {2:.4f}, Error: {3:.2f}%".format(epoch, i, training_loss, eval_error * 100.0))

        # Save batch marker
        f['.']['time_markers']['minibatch'][epoch] = np.float32(batch_count)

        # Validation
        validation_loss = 0
        validation_error = 0
        for j in range(valid_batch_count):
            # Read a mini batch from the validation data file
            data = cntk_valid_src.next_minibatch(batch_size, input_map=valid_map)

            # Valid a batch
            cntk_trainer.test_minibatch(data)
            validation_loss += cntk_trainer.previous_minibatch_loss_average * len(data)
            validation_error += cntk_trainer.previous_minibatch_evaluation_average * len(data)

        validation_loss /= len(x_valid)
        validation_error /= len(x_valid)

        # Save validation loss for the whole epoch
        f['.']['cost']['loss'][epoch] = np.float32(validation_loss)
        f['.']['accuracy']['valid'][epoch] = np.float32((1.0 - validation_error) * 100.0)

    # Save related params
    f['.']['time']['train']['end_time'][0] = time.time() # Save training time
    f['.']['config'].attrs["total_minibatches"] = batch_count
    f['.']['time_markers'].attrs['minibatches_complete'] = batch_count

    # Testing
    test_loss = 0
    test_error = 0
    for j in range(test_batch_count):
        # Read a mini batch from the validation data file
        data = cntk_test_src.next_minibatch(batch_size, input_map=test_map)

        # Valid a batch
        cntk_trainer.test_minibatch(data)
        test_loss += cntk_trainer.previous_minibatch_loss_average * len(data)
        test_error += cntk_trainer.previous_minibatch_evaluation_average * len(data)

    test_loss /= len(testImages)
    test_error /= len(testImages)

    f['.']['infer_acc']['accuracy'][0] = np.float32((1.0 - test_error) * 100.0)
    print("Accuracy score is %f" % (1.0 - test_error))

except KeyboardInterrupt:
    pass
except Exception as e:
    raise e
finally:
    print("Close file descriptor")
    f.close()



# # Validation and testing configuration
# cntk_valid_config = CrossValidationConfig(
# 	minibatch_source = cntk_valid_src,
# 	frequency = (1, DataUnit.sweep),
# 	minibatch_size = batch_size,
#     model_inputs_to_streams = valid_map,
#     max_samples = len(x_valid),
#     criterion = (cntk_cost, cntk_error))
# cntk_test_config = TestConfig(
# 	minibatch_source = cntk_test_src,
# 	minibatch_size = batch_size,
# 	model_inputs_to_streams = test_map,
# 	criterion = (cntk_cost, cntk_error))

# # Start training
# training_session(
#         trainer = cntk_trainer,
#         mb_source = cntk_train_src,
#         mb_size = batch_size,
#         model_inputs_to_streams = train_map,
#         max_samples = len(x_train) * epoch_num,
#         progress_frequency = len(x_train),
#         cv_config = cntk_valid_config,
#         test_config = cntk_test_config).train()
