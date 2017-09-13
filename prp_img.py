from PIL import Image
from six.moves import cPickle
import csv, time, os.path
import matplotlib.pyplot as plt

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath, size, training=True):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 43 classes
    if training:
        for c in range(0,43):
            prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
            gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader) # skip header
            # loop over all images in current annotations file
            for row in gtReader:
#                 image = Image.open(prefix + row[0]).convert('L') # Load an image and convert to grayscale
                image = Image.open(prefix + row[0])
                box = (int(row[3]), int(row[4]), int(row[5]), int(row[6])) # Specify ROI box
                image = image.crop(box) # Crop the ROI
                image = image.resize(size) # Resize images
                images.append(np.asarray(image).astype('uint8')) # the 1th column is the filename, while 3,4,5,6 are the vertices of ROI
                labels.append(int(row[7])) # the 8th column is the label
            gtFile.close()
    else:
        gtFile = open(rootpath + "/../../GT-final_test.csv") # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
#             image = Image.open(rootpath + '/' + row[0]).convert('L') # Load an image and convert to grayscale
            image = Image.open(rootpath + '/' + row[0]) # Color version
            box = (int(row[3]), int(row[4]), int(row[5]), int(row[6])) # Specify ROI box
            image = image.crop(box) # Crop the ROI
            image = image.resize(size) # Resize images
            images.append(np.asarray(image).astype('uint8')) # the 1th column is the filename, while 3,4,5,6 are the vertices of ROI
            labels.append(int(row[7])) # the 8th column is the label
        gtFile.close()
        
    return images, labels

def getImageSets(root, resize_size):
	train_dir = root + "/Final_Training/Images"
	test_dir = root + "/Final_Test/Images"

	## If pickle file exists, read the file
	if os.path.isfile(root + "/processed_images.pkl"):
		f = open(root + "/processed_images.pkl", 'rb')
		trainImages = cPickle.load(f, encoding="latin1")
		trainLabels = cPickle.load(f, encoding="latin1")
		testImages = cPickle.load(f, encoding="latin1")
		testLabels = cPickle.load(f, encoding="latin1")
		f.close()
	## Else, read images and write to the pickle file
	else:
		start = time.time()
		trainImages, trainLabels = readTrafficSigns(train_dir, resize_size)
		print("Training Image preprocessing finished in {:.2f} seconds".format(time.time() - start))

		start = time.time()
		testImages, testLabels = readTrafficSigns(test_dir, resize_size, False)
		print("Testing Image preprocessing finished in {:.2f} seconds".format(time.time() - start))
		
		f = open(root + "/processed_images.pkl", 'wb')

		for obj in [trainImages, trainLabels, testImages, testLabels]:
		    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()

	print(trainImages[42].shape)
	# plt.imshow(trainImages[42])
	# plt.show()

	print(testImages[21].shape)
	# plt.imshow(trainImages[21])
	# plt.show()

	return trainImages, trainLabels, testImages, testLabels
