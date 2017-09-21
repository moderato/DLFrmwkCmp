import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import model_selection as ms
from sys import platform
from prp_img import getImageSets
import DLHelper

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

import mxnet as mx
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

class MxCustomInit(mx.initializer.Initializer):
    def __init__(self, idict):
        super(MxCustomInit, self).__init__()
        self.dict = idict
        np.random.seed(seed=1)

    def _init_weight(self, name, arr):
        if name in self.dict.keys():
            dictPara = self.dict[name]
            for(k, v) in dictPara.items():
                arr = np.random.normal(0, v, size=arr.shape)

    def _init_bias(self, name, arr):
        if name in self.dict.keys():
            dictPara = self.dict[name]
            for(k, v) in dictPara.items():
                arr[:] = v

class MxBatchCallback(object):
    def __init__(self, f, batch_size, auto_reset=True):
        self.batch_size = batch_size
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.batch_count = 0
        self.auto_reset = auto_reset
        self.f = f

    def __call__(self, param):
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count
        self.batch_count += 1

        if self.init:
            batch_time = time.time() - self.tic # self.tic: batch start time
            self.f['.']['time']['train_batch'][self.batch_count-1] = batch_time

            # param.epoch: Epoch index
            # param.eval_metric: Real time metrics (Use param.eval_metric.get_name_value() to get it)
            # param.nbatch: Current batch count (in one epoch)
            # param.locals: Miscellaneous

            self.f['.']['cost']['train'][self.batch_count-1] = np.float32(param.eval_metric.get_name_value()[1][1])
            self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


class MxEpochCallback(object):
    def __init__(self, prefix, batch_size, epoch_num, auto_reset=True):
        self.prefix = prefix
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset

    def __call__(self, iter_no, sym, arg, aux):
        if (iter_no + 1) == self.epoch_num:
            mx.model.save_checkpoint(self.prefix, iter_no+1, sym, arg, aux)


# Prepare image sets
# batch size = (batch, 3, size_x, size_y)
mx_train_x = mx.nd.array([i.swapaxes(0,2).astype("float32")/255 for i in x_train])
mx_valid_x = mx.nd.array([i.swapaxes(0,2).astype("float32")/255 for i in x_valid])
mx_test_x = mx.nd.array([i.swapaxes(0,2).astype("float32")/255 for i in testImages])
mx_train_y = mx.nd.array(y_train, dtype=np.float32) # No need of one_hot
mx_valid_y = mx.nd.array(y_valid, dtype=np.float32)
mx_test_y = mx.nd.array(testLabels, dtype=np.float32)

# The iterators have input name of 'data' and output name of 'softmax_label' if not particularly specified
mx_train_set = mx.io.NDArrayIter(mx_train_x, mx_train_y, batch_size, shuffle=True)
mx_valid_set = mx.io.NDArrayIter(mx_valid_x, mx_valid_y, batch_size)
mx_test_set = mx.io.NDArrayIter(mx_test_x, mx_test_y, batch_size)

# Print the shape and type of training set lapel
# mx_train_set.provide_label

# Construct the network
data = mx.sym.Variable('data')
mx_conv1 = mx.sym.Convolution(data = data, name='mx_conv1', num_filter=64, kernel=(5,5), stride=(2,2))
mx_act1 = mx.sym.Activation(data = mx_conv1, name='mx_relu1', act_type="relu")
mx_mp1 = mx.sym.Pooling(data = mx_act1, name = 'mx_pool1', kernel=(2,2), stride=(2,2), pool_type='max')
mx_conv2 = mx.sym.Convolution(data = mx_mp1, name='mx_conv2', num_filter=512, kernel=(3,3), stride=(1,1), pad=(1,1))
mx_act2 = mx.sym.Activation(data = mx_conv2, name='mx_relu2', act_type="relu")
mx_mp2 = mx.sym.Pooling(data = mx_act2, name = 'mx_pool2', kernel=(2,2), stride=(2,2), pool_type='max')
mx_fl = mx.sym.Flatten(data = mx_mp2, name="mx_flatten")
mx_fc1 = mx.sym.FullyConnected(data = mx_fl, name='mx_fc1', num_hidden=2048)
mx_drop = mx.sym.Dropout(data = mx_fc1, name='mx_dropout', p=0.5)
mx_fc2 = mx.sym.FullyConnected(data = mx_drop, name='mx_fc2', num_hidden=43)
mx_softmax = mx.sym.SoftmaxOutput(data = mx_fc2, name ='softmax')

# Print the names of arguments in the model
# mx_softmax.list_arguments() # Make sure the input and the output names are consistent of those in the iterator!!

# Print the size of the model
# mx_softmax.infer_shape(data=(1,3,49,49))

# Draw the network
# mx.viz.plot_network(mx_softmax, shape={"data":(batch_size, 3, resize_size[0], resize_size[1])})

# Initialization params
mx_nor_dict = {'normal': 0.01}
mx_cons_dict = {'constant': 0.0}
mx_init_dict = {}
for layer in mx_softmax.list_arguments():
    hh = layer.split('_')
    if hh[-1] == 'weight':
        mx_init_dict[layer] = mx_nor_dict
    elif hh[-1] == 'bias':
        mx_init_dict[layer] = mx_cons_dict
# print(mx_init_dict)

# create a trainable module on CPU
mx_model = mx.mod.Module(context = mx.cpu(), symbol = mx_softmax)

max_total_batch = (len(x_train) / batch_size + 1) * epoch_num
filename = root + "/saved_data/callback_data_mxnet_cpu.h5"
f = DLHelper.init_h5py(filename, epoch_num, max_total_batch)

try:
    # Train the model
    # Currently no solution to reproducibility. Eyes on issue 47.
    mx_model.fit(mx_train_set, # train data
                 eval_data = mx_valid_set, # validation data
                 num_epoch = epoch_num,
                 # initializer = MxCustomInit(mx_init_dict), # Bugs. Don't use it now.
                 eval_metric = ['acc', 'ce'], # Calculate accuracy and cross-entropy
                 optimizer = 'sgd',
                 optimizer_params = {'learning_rate': 0.1, 'momentum': 0.9},
                 epoch_end_callback = MxEpochCallback("try", batch_size, epoch_num),
                 batch_end_callback = MxBatchCallback(f, batch_size))
except KeyboardInterrupt:
    pass
except Exception as e:
    raise e
finally:
    f.close()


score = mx_model.score(mx_test_set, ['acc'])
print("Accuracy score is %f" % (score[0][1])) # 0 for acc index in metric list, 1 for 
