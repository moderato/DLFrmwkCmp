import sys
import caffe
import numpy as np
import cv2
import DLHelper
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing import image

def load_weights_from_hdf5(config_filename, weight_filename):

    js = open(config_filename, 'r').readlines()[0]

    keras_model = model_from_json(js)
    
    keras_model.load_weights(weight_filename)
    keras_model.summary()

    layers = keras_model.layers

    return keras_model

def convert_filter(numpy_filter_weight):
    return np.transpose(numpy_filter_weight,(3,2,1,0))

# root = "/Users/moderato/Documents/Libraries/caffe/models/"
# size = (227, 227)

# pimga = cv2.imread(root + "cat.jpg")
# pimga = cv2.resize(pimga, size)
# nimga = np.array(pimga).reshape(1,size[0],size[1],3).transpose(0,3,1,2)

# net = caffe.Net(root + "bvlc_reference_caffenet/deploy.prototxt", root + "bvlc_reference_caffenet.caffemodel", caffe.TEST)


root = "/Users/moderato/Downloads/"
resize_size = (48, 48)
dataset = "GT"

root, trainImages, trainLabels, testImages, testLabels, class_num = DLHelper.getImageSets(root, resize_size, dataset=dataset, printing=False)

keras_model = load_weights_from_hdf5('./saved_models/keras_tensorflow_config.json', \
        './saved_models/keras_tensorflow_weights.hdf5')

# first_layer = keras_model.layers[0]
# w, b = first_layer.get_weights()
# w = convert_filter(w)
# print(w)



# keras_test_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in testImages])

# keras_predictions = [np.argmax(keras_model.predict(np.expand_dims(feature[:, :, [2, 1, 0]], axis=0))) for feature in keras_test_x[1:10]]

# print(keras_predictions)



# from keras import backend as K

# inp = keras_model.input                                           # input placeholder
# outputs = [layer.output for layer in keras_model.layers]          # all layer outputs
# functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

# # Testing
# test = np.expand_dims(keras_test_x[0], axis=0)
# layer_outs = [func([test, 1.]) for func in functors]
# count = 0
# for o in layer_outs:
# 	print("**************** {} ***************".format(count))
# 	print(o)
# 	count += 1
# 	print("***********************************")


net = caffe.Classifier("./converted_model/idsia.prototxt", "./converted_model/idsia.caffemodel", caffe.TEST)
# # w = net.params['conv1'][0].data
# # print("*******")
# # print(w)

nimga = np.array(testImages[0])[:, :, [2, 1, 0]]
shape = nimga.shape
nimga = nimga.reshape(1, shape[0], shape[1], shape[2]).transpose(0, 3, 1, 2)


# nimgas = [np.array(img).reshape(1, resize_size[0], resize_size[1], 3).transpose(0, 3, 1, 2) / 255.0 for img in testImages[1:20]]
# out = net.predict(nimgas)
out = net.forward_all(**{"data": nimga})
# # print(net.blobs['data'].data)

# # # print(out)
# # print(testLabels[1:20])
# for o in out:
print("Predicted class is #{}.".format(out['prob'].argmax()))