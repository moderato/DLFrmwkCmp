import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import model_selection as ms
from sys import platform
import DLHelper
from timeit import default_timer

if platform == "darwin":
    root = "/Users/moderato/Downloads/GTSRB/try"
else:
    root = "/home/zhongyilin/Desktop/GTSRB/try"
print(root)
resize_size = (48, 48)
trainImages, trainLabels, testImages, testLabels = DLHelper.getImageSets(root, resize_size)
x_train, x_valid, y_train, y_valid = ms.train_test_split(trainImages, trainLabels, test_size=0.2, random_state=542)

epoch_num = 25
batch_size = 64

import cntk as C
from cntk.learners import momentum_sgd as cntk_SGD
from cntk import cross_entropy_with_softmax as cntk_softmax, classification_error as cntk_error
from cntk.io import MinibatchSource

if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

def constructCNN(cnn_type='self'):
	net = None
	if cnn_type == 'idsia':
	    with C.layers.default_options(init=C.normal(0.01), activation=C.relu):
	    	model = C.layers.Sequential([	            
	    		C.layers.Convolution((3,3), 100, pad=False),
	    		C.layers.MaxPooling((2,2), strides=(2,2)),
	    		C.layers.Convolution((4,4), 150, pad=False),
	    		C.layers.MaxPooling((2,2), strides=(2,2)),
	    		C.layers.Convolution((3,3), 250, pad=False),
	    		C.layers.MaxPooling((2,2), strides=(2,2)),

	            C.layers.Dense(200),
	            C.layers.Dense(43, activation=None)
	        ])
	elif cnn_type == 'self':
		with C.layers.default_options(init=C.normal(0.01), activation=C.relu):
	    	model = C.layers.Sequential([	            
	    		C.layers.Convolution((5,5), 64, pad=True, strides=2),
	    		C.layers.MaxPooling((2,2), strides=(2,2)),
	    		C.layers.Convolution((3,3), 512, pad=True),
	    		C.layers.MaxPooling((2,2), strides=(2,2)),

	            C.layers.Dense(2048),
	            C.layers.Dropout(0.5),
	            C.layers.Dense(43, activation=None)
	        ])
    
    return model


cntk_input_x = C.input_variable((3, resize_size[0], resize_size[1]))
cntk_input_y = C.input_variable((43))

cntk_model = constructCNN('idsia')(cntk_input_x)

cntk_cost = cntk_softmax(cntk_model, cntk_input_y)
cntk_error = cntk_error(cntk_model, cntk_input_y)