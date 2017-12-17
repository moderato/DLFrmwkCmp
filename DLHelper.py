import h5py, cv2
from six.moves import cPickle
import csv, time, os.path
import matplotlib.pyplot as plt
import numpy as np

# function to process a single image
def processImage(prefix, size, gtReader, proc_type=None):
    images = []
    labels = []

    for row in gtReader:
        image = cv2.imread(prefix + row[0])
        image = image[...,::-1] # BGR to RGB
        image = image[int(row[4]):int(row[6]), int(row[3]):int(row[5])] # Crop the ROI
        image = cv2.resize(image, size) # Resize images 
        if proc_type is None:
            pass
        elif proc_type == "CLAHE" or proc_type == "clahe":
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) # BGR to Lab space
            tmp = np.zeros((lab.shape[0],lab.shape[1]), dtype=lab.dtype)
            tmp[:,:] = lab[:,:,0] # Get the light channel of LAB space
            clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(4,4)) # Create CLAHE object
            light = clahe.apply(tmp) # Apply to the light channel
            lab[:,:,0] = light # Merge back
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) # LAB to RGB
        elif proc_type == "1sigma" or proc_type == "2sigma":
            R, G, B = image[:,:,0], image[:,:,1], image[:,:,2] # RGB channels
            if proc_type == "1sigma":
                param = 1
            else: # "2sigma"
                param = 2
            image[:,:,0] = cv2.normalize(R, None, R.mean() - param * R.std(), R.mean() + param * R.std(), cv2.NORM_MINMAX)
            image[:,:,1] = cv2.normalize(G, None, G.mean() - param * G.std(), G.mean() + param * G.std(), cv2.NORM_MINMAX)
            image[:,:,2] = cv2.normalize(B, None, B.mean() - param * B.std(), B.mean() + param * B.std(), cv2.NORM_MINMAX)

        images.append(image) # Already uint8
        labels.append(int(row[7])) # the 8th column is the label

    return images, labels

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns_GT(rootpath, size, process=None, training=True):
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
            imgs, lbls = processImage(prefix, size, gtReader, process)
            images = images + imgs
            labels = labels + lbls
            gtFile.close()
    else:
        gtFile = open(rootpath + "/../../GT-final_test.csv") # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        imgs, lbls = processImage(rootpath + '/', size, gtReader, process)
        images = images + imgs
        labels = labels + lbls
        gtFile.close()

    return images, labels

def readTrafficSigns_Belgium(rootpath, size, process=None, training=True):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all classes
    for c in range(0,62):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        imgs, lbls = processImage(prefix, size, gtReader, process)
        images = images + imgs
        labels = labels + lbls
        gtFile.close()

    return images, labels

def getDirFuncClassNum(root, dataset="GT"):
    train_dir, test_dir, readTrafficSigns = None, None, None
    class_num = -1
    if dataset == "GT":
        root += "/GTSRB/try"
        train_dir = root + "/Final_Training/Images"
        test_dir = root + "/Final_Test/Images"
        readTrafficSigns = readTrafficSigns_GT
        class_num = 43
    elif dataset == "Belgium":
        root += "/BelgiumTSC"
        train_dir = root + "/Training"
        test_dir = root + "/Testing"
        readTrafficSigns = readTrafficSigns_Belgium
        class_num = 62
    else:
        raise Exception("")

    return root, train_dir, test_dir, readTrafficSigns, class_num


def getImageSets(root, resize_size, dataset="GT", process=None, printing=True):
    root, train_dir, test_dir, readTrafficSigns, class_num = getDirFuncClassNum(root, dataset)
    trainImages, trainLabels, testImages, testLabels = None, None, None, None

    ## If pickle file exists, read the file
    if os.path.isfile(root + "/processed_images_{}_{}_{}_{}.pkl".format(resize_size[0], resize_size[1], dataset, (process if (process is not None) else "original"))):
        f = open(root + "/processed_images_{}_{}_{}_{}.pkl".format(resize_size[0], resize_size[1], dataset, (process if (process is not None) else "original")), 'rb')
        trainImages = cPickle.load(f, encoding="latin1")
        trainLabels = cPickle.load(f, encoding="latin1")
        testImages = cPickle.load(f, encoding="latin1")
        testLabels = cPickle.load(f, encoding="latin1")
        f.close()
    ## Else, read images and write to the pickle file
    else:
        start = time.time()
        trainImages, trainLabels = readTrafficSigns(train_dir, resize_size, process, True)
        print("Training Image preprocessing finished in {:.2f} seconds".format(time.time() - start))

        start = time.time()
        testImages, testLabels = readTrafficSigns(test_dir, resize_size, process, False)
        print("Testing Image preprocessing finished in {:.2f} seconds".format(time.time() - start))
        
        f = open(root + "/processed_images_{}_{}_{}_{}.pkl".format(resize_size[0], resize_size[1], dataset, (process if (process is not None) else "original")), 'wb')

        for obj in [trainImages, trainLabels, testImages, testLabels]:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    if printing:
        print(trainImages[42].shape)
        plt.imshow(trainImages[42])
        plt.show()

        print(testImages[21].shape)
        plt.imshow(trainImages[21])
        plt.show()

    return root, trainImages, trainLabels, testImages, testLabels, class_num

def init_h5py(filename, epoch_num, max_total_batch):
    f = h5py.File(filename, 'w')
        
    try:
        # config group for some common params
        config = f.create_group('config')
        config.attrs["total_epochs"] = epoch_num

        # cost group for training and validation cost
        cost = f.create_group('cost')
        loss = cost.create_dataset('loss', (epoch_num,))
        loss.attrs['time_markers'] = 'epoch_freq'
        loss.attrs['epoch_freq'] = 1
        train = cost.create_dataset('train', (max_total_batch,)) # Set size to maximum theoretical value
        train.attrs['time_markers'] = 'minibatch'

        # time group for batch and epoch time
        t = f.create_group('time')
        loss = t.create_dataset('loss', (epoch_num,))
        train = t.create_group('train')
        start_time = train.create_dataset("start_time", (1,))
        start_time.attrs['units'] = 'seconds'
        end_time = train.create_dataset("end_time", (1,))
        end_time.attrs['units'] = 'seconds'
        train_batch = t.create_dataset('train_batch', (max_total_batch,)) # Same as above

        # accuracy group for training and validation accuracy
        acc = f.create_group('accuracy')
        acc_v = acc.create_dataset('valid', (epoch_num,))
        acc_v.attrs['time_markers'] = 'epoch_freq'
        acc_v.attrs['epoch_freq'] = 1
        acc_t = acc.create_dataset('train', (max_total_batch,))
        acc_t.attrs['time_markers'] = 'minibatch'

        # Mark which batches are the end of an epoch
        time_markers = f.create_group('time_markers')
        time_markers.attrs['epochs_complete'] = epoch_num
        train_batch = time_markers.create_dataset('minibatch', (epoch_num,))

        # Inference accuracy
        infer = f.create_group('infer_acc')
        infer_acc = infer.create_dataset('accuracy', (1,))

    except Exception as e:
        f.close() # Avoid hdf5 runtime error or os error
        raise e # Catch the exception to close the file, then raise it to stop the program

    return f
