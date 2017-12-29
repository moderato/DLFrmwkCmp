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

root, trainImages, trainLabels, testImages, testLabels, class_num = DLHelper.getImageSets(root, resize_size, dataset=dataset, printing=True)
x_train, x_valid, y_train, y_valid = ms.train_test_split(trainImages, trainLabels, test_size=0.2, random_state=542)

epoch_num = 25
batch_size = 64

import cntk as C
from cntk.learners import momentum_sgd as cntk_SGD
from cntk import cross_entropy_with_softmax as cntk_softmax, classification_error as cntk_error
from cntk.io import MinibatchSourceFromData
from cntk.logging import ProgressPrinter
from cntk.train.training_session import *

if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

def constructCNN(layer_name_prefix, cnn_type='self'):
	model = None
	if cnn_type == 'idsia':
	    with C.layers.default_options(init=C.normal(0.01), activation=C.relu):
	    	model = C.layers.Sequential([	            
	    		C.layers.Convolution((3,3), strides=(1,1), num_filters=100, pad=False,\
	    			name=layer_name_prefix+"conv1"),
	    		C.layers.MaxPooling((2,2), strides=(2,2), name=layer_name_prefix+"pool1"),
	    		C.layers.Convolution((4,4), strides=(1,1), num_filters=150, pad=False,\
	    			name=layer_name_prefix+"conv2"),
	    		C.layers.MaxPooling((2,2), strides=(2,2), name=layer_name_prefix+"pool2"),
	    		C.layers.Convolution((3,3), strides=(1,1), num_filters=250, pad=False,\
	    			name=layer_name_prefix+"conv3"),
	    		C.layers.MaxPooling((2,2), strides=(2,2), name=layer_name_prefix+"pool3"),

	            C.layers.Dense(200, name=layer_name_prefix+"fc1"),
	            C.layers.Dense(class_num, activation=None, name=layer_name_prefix+"fc2") # Leave the softmax for now
	        ])
	elif cnn_type == 'self':
		with C.layers.default_options(init=C.normal(0.01), activation=C.relu):
	    	model = C.layers.Sequential([	            
	    		C.layers.Convolution((5,5), strides=(2,2), num_filters=64, pad=True,\
	    			name=layer_name_prefix+"conv1"),
	    		C.layers.MaxPooling((2,2), strides=(2,2), name=layer_name_prefix+"pool1"),
	    		C.layers.Convolution((3,3), strides=(1,1), num_filters=256, pad=True,\
	    			name=layer_name_prefix+"conv2"),
	    		C.layers.MaxPooling((2,2), strides=(2,2), name=layer_name_prefix+"pool2"),

	            C.layers.Dense(2048, name=layer_name_prefix+"fc1"),
	            C.layers.Dropout(0.5, name=layer_name_prefix+"dropout1"),
	            C.layers.Dense(class_num, activation=None, name=layer_name_prefix+"fc2") # Leave the softmax for now
	        ])
    
    return model

cntk_input_x = C.input_variable((3, resize_size[0], resize_size[1]), np.float32)
cntk_input_y = C.input_variable(class_num, np.float32)
cntk_model = constructCNN('idsia')(cntk_input_x)
cntk_cost = cntk_softmax(cntk_model, cntk_input_y)
cntk_error = cntk_error(cntk_model, cntk_input_y)


progress_writers = [ProgressPrinter(
        tag='Training',
        num_epochs=epoch_num)]

cntk_learner = cntk_SGD(cntk_model.parameters, lr=0.01, momentum=0.9)
cntk_trainer = C.Trainer(cntk_model, (cntk_cost, cntk_error), [cntk_learner], [progress_writers])

cntk_train_src = C.io.MinibatchSourceFromData(dict(x=C.Value(x_train), y=C.Value(y_train)))
cntk_valid_src = C.io.MinibatchSourceFromData(dict(x=C.Value(x_valid), y=C.Value(y_valid)))
cntk_test_src = C.io.MinibatchSourceFromData(dict(x=C.Value(testImages), y=C.Value(testLabels)))

input_map = {
    feature: cntk_train_src.streams.features,
    label: cntk_train_src.streams.labels
}

training_session(
        trainer = cntk_trainer,
        mb_source = cntk_train_src,
        mb_size = batch_size,
        model_inputs_to_streams = input_map,
        max_samples = len(x_train) * epoch_num,
        progress_frequency = len(x_train),
        cv_config = CrossValidationConfig(cntk_valid_src, batch_size),
        test_config = TestConfig(cntk_test_src, batch_size)
    ).train()
