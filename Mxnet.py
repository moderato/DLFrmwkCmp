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

epoch_num = 5
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

# Prepare image sets
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

# Train the model
# Currently no solution to reproducibility. Eyes on issue 47.
mx_model.fit(mx_train_set, # train data
             eval_data = mx_valid_set, # validation data
             num_epoch = epoch_num,
             initializer = MxCustomInit(mx_init_dict),
             optimizer = 'sgd',
             optimizer_params = {'learning_rate': 0.1, 'momentum': 0.9},
             eval_metric ='acc', # report accuracy during training
             batch_end_callback = mx.callback.Speedometer(batch_size, 10)) # output progress for each 10 data batches

score = mx_model.score(mx_test_set, ['acc'])
print("Accuracy score is %f" % (score[0][1]))
