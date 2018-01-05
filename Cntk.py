import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
import time, sys, DLHelper

print("**********************************")
print("Training on CNTK")
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

root, trainImages, trainLabels, testImages, testLabels, class_num = DLHelper.getImageSets(root, resize_size, dataset=dataset, process=process, printing=printing)
x_train, x_valid, y_train, y_valid = ms.train_test_split(trainImages, trainLabels, test_size=0.2, random_state=542)

_ = DLHelper.create_dir(root, ["saved_data", "saved_models"], network_type, backends)

import cntk as C
import cntk_resnet
from cntk.learners import momentum_sgd as SGD
from cntk import cross_entropy_with_softmax as Softmax, classification_error as ClassificationError
from cntk.io import MinibatchSourceFromData
from cntk.logging import ProgressPrinter
from cntk.train.training_session import *
from cntk.initializer import he_normal
from timeit import default_timer

backend = 'cpu'
if C.device.use_default_device().type() == 0:
    print('running on CPU')
else:
    print('running on GPU')
    backend = 'gpu'

def constructCNN(cntk_input, cnn_type='self'):
    model = None
    if cnn_type == 'idsia':
        with C.layers.default_options(activation=C.relu):
            model = C.layers.Sequential([
                C.layers.Convolution((3,3), strides=(1,1), num_filters=100, pad=False,
                    init=he_normal(), name="cntk_conv1"),
                C.layers.MaxPooling((2,2), strides=(2,2), name="cntk_pool1"),
                C.layers.Convolution((4,4), strides=(1,1), num_filters=150, pad=False,
                    init=he_normal(), name="cntk_conv2"),
                C.layers.MaxPooling((2,2), strides=(2,2), name="cntk_pool2"),
                C.layers.Convolution((3,3), strides=(1,1), num_filters=250, pad=False,
                    init=he_normal(), name="cntk_conv3"),
                C.layers.MaxPooling((2,2), strides=(2,2), name="cntk_pool3"),

                C.layers.Dense(200, init=he_normal(), name="cntk_fc1"),
                C.layers.Dense(class_num, activation=None, init=he_normal(), name="cntk_fc2") # Leave the softmax for now
            ])(cntk_input)
    elif cnn_type == 'self':
        with C.layers.default_options(activation=C.relu):
            model = C.layers.Sequential([
                C.layers.Convolution((5,5), strides=(2,2), num_filters=64, pad=True,
                    init=he_normal(), name="cntk_conv1"),
                C.layers.MaxPooling((2,2), strides=(2,2), name="cntk_pool1"),
                C.layers.Convolution((3,3), strides=(1,1), num_filters=256, pad=True,
                    init=he_normal(), name="cntk_conv2"),
                C.layers.MaxPooling((2,2), strides=(2,2), name="cntk_pool2"),

                C.layers.Dense(2048, init=he_normal(), name="cntk_fc1"),
                C.layers.Dropout(0.5, name="cntk_dropout1"),
                C.layers.Dense(class_num, activation=None, init=he_normal(), name="cntk_fc2") # Leave the softmax for now
            ])(cntk_input)
    elif cnn_type == "resnet-56":
        model = cntk_resnet.create_model(cntk_input, 9, class_num) # 6*9 + 2 = 56
    elif cnn_type == "resnet-32":
        model = cntk_resnet.create_model(cntk_input, 5, class_num) # 6*5 + 2 = 32

    return model

# Construct model, io and metrics
cntk_input = C.input_variable((3, resize_size[0], resize_size[1]), np.float32)
cntk_output = C.input_variable((class_num), np.float32)
cntk_model = constructCNN(cntk_input, network_type)
cntk_cost = Softmax(cntk_model, cntk_output)
cntk_error = ClassificationError(cntk_model, cntk_output)


# Construct data
cntk_train_x = np.ascontiguousarray(np.vstack([np.expand_dims(x, axis=0).transpose([0,3,1,2]).astype('float32')/255 for x in x_train]), dtype=np.float32)
cntk_valid_x = np.ascontiguousarray(np.vstack([np.expand_dims(x, axis=0).transpose([0,3,1,2]).astype('float32')/255 for x in x_valid]), dtype=np.float32)
cntk_test_x = np.ascontiguousarray(np.vstack([np.expand_dims(x, axis=0).transpose([0,3,1,2]).astype('float32')/255 for x in testImages]), dtype=np.float32)

cntk_train_y = C.one_hot(C.input_variable(1), class_num, sparse_output=False)(np.expand_dims(np.array(y_train, dtype='f'), axis=1))
cntk_valid_y = C.one_hot(C.input_variable(1), class_num, sparse_output=False)(np.expand_dims(np.array(y_valid, dtype='f'), axis=1))
cntk_test_y = C.one_hot(C.input_variable(1), class_num, sparse_output=False)(np.expand_dims(np.array(testLabels, dtype='f'), axis=1))


# Trainer and mb source
cntk_learner = SGD(cntk_model.parameters, lr=0.01, momentum=0.9)
cntk_trainer = C.Trainer(cntk_model, (cntk_cost, cntk_error), cntk_learner)
cntk_train_src = C.io.MinibatchSourceFromData(dict(x=C.Value(cntk_train_x), y=C.Value(cntk_train_y)), max_samples=len(cntk_train_x))
cntk_valid_src = C.io.MinibatchSourceFromData(dict(x=C.Value(cntk_valid_x), y=C.Value(cntk_valid_y)), max_samples=len(cntk_valid_x))
cntk_test_src = C.io.MinibatchSourceFromData(dict(x=C.Value(cntk_test_x), y=C.Value(cntk_test_y)), max_samples=len(cntk_test_x))

# Mapping for training, validation and testing
def getMap(src, bs):
    batch = src.next_minibatch(bs)
    return {
        cntk_input: batch[src.streams['x']],
        cntk_output: batch[src.streams['y']]
    }

# Create log file
train_batch_count = len(x_train) // batch_size + 1
valid_batch_count = len(x_valid) // batch_size + 1
test_batch_count = len(testImages) // batch_size + 1
filename = "{}saved_data/{}/{}/callback_data_cntk_{}.h5".format(root, network_type, backend, dataset)
f = DLHelper.init_h5py(filename, epoch_num, train_batch_count * epoch_num)

# Start training
try:
    batch_count = 0
    f['.']['time']['train']['start_time'][0] = time.time()

    # Each epoch
    for epoch in range(0, epoch_num):
        cntk_train_src.restore_from_checkpoint({'cursor': 0, 'total_num_samples': 0})
        cntk_valid_src.restore_from_checkpoint({'cursor': 0, 'total_num_samples': 0})
        cntk_test_src.restore_from_checkpoint({'cursor': 0, 'total_num_samples': 0})

        # Each batch
        for i in range(train_batch_count):
            batch_count += 1

            # Read a mini batch from the training data file
            data = getMap(cntk_train_src, batch_size)

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

            if i % 30 == 0: # Print per 50 batches
                print("Epoch: {0}, Minibatch: {1}, Loss: {2:.4f}, Error: {3:.2f}%".format(epoch, i, training_loss, eval_error * 100.0))

        # Save batch marker
        f['.']['time_markers']['minibatch'][epoch] = np.float32(batch_count)

        # Validation
        validation_loss = 0
        validation_error = 0
        for j in range(valid_batch_count):
            # Read a mini batch from the validation data file
            data = getMap(cntk_valid_src, batch_size)

            # Valid a batch
            batch_x, batch_y = data[cntk_input].asarray(), data[cntk_output].asarray()
            validation_loss += cntk_cost(batch_x, batch_y).sum()
            validation_error += cntk_trainer.test_minibatch(data) * len(batch_x)

        validation_loss /= len(x_valid)
        validation_error /= len(x_valid)

        # Save validation loss for the whole epoch
        f['.']['cost']['loss'][epoch] = np.float32(validation_loss)
        f['.']['accuracy']['valid'][epoch] = np.float32((1.0 - validation_error) * 100.0)
        print("[Validation]")
        print("Epoch: {0}, Loss: {1:.4f}, Error: {2:.2f}%\n".format(epoch, validation_loss, validation_error * 100.0))

    # Save related params
    f['.']['time']['train']['end_time'][0] = time.time() # Save training time
    f['.']['config'].attrs["total_minibatches"] = batch_count
    f['.']['time_markers'].attrs['minibatches_complete'] = batch_count

    # Testing
    test_error = 0
    for j in range(test_batch_count):
        # Read a mini batch from the validation data file
        data = getMap(cntk_test_src, batch_size)

        # Valid a batch
        test_error += cntk_trainer.test_minibatch(data) * len(data[cntk_input].asarray())

    test_error /= len(testImages)

    f['.']['infer_acc']['accuracy'][0] = np.float32((1.0 - test_error) * 100.0)
    print("Accuracy score is %f" % (1.0 - test_error))

    cntk_model.save("{}saved_model/{}/{}/cntk_{}.pth".format(root, network_type, backend, dataset))

except KeyboardInterrupt:
    pass
except Exception as e:
    raise e
finally:
    print("Close file descriptor")
    f.close()



# # Validation and testing configuration
# cntk_valid_config = CrossValidationConfig(
#     minibatch_source = cntk_valid_src,
#     frequency = (1, DataUnit.sweep),
#     minibatch_size = batch_size,
#     model_inputs_to_streams = valid_map,
#     max_samples = len(x_valid),
#     criterion = (cntk_cost, cntk_error))
# cntk_test_config = TestConfig(
#     minibatch_source = cntk_test_src,
#     minibatch_size = batch_size,
#     model_inputs_to_streams = test_map,
#     criterion = (cntk_cost, cntk_error))

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
