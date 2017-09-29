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
resize_size = (49, 49)
trainImages, trainLabels, testImages, testLabels = DLHelper.getImageSets(root, resize_size)
x_train, x_valid, y_train, y_valid = ms.train_test_split(trainImages, trainLabels, test_size=0.2, random_state=542)

epoch_num = 10
batch_size = 128

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import torch.nn.init as torch_init
from torchvision import datasets, transforms
from torch.autograd import Variable

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Build model
        self.conv = torch.nn.Sequential()
        self.conv.add_module("torch_conv1", torch.nn.Conv2d(3, 64, kernel_size=(5, 5), stride=2))
        self.conv.add_module("torch_pool1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("torch_relu1", torch.nn.ReLU())
        self.conv.add_module("torch_conv2", torch.nn.Conv2d(64, 256, kernel_size=(3, 3), stride=1, padding=1))
        self.conv.add_module("torch_pool2", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("torch_relu2", torch.nn.ReLU())
        self.conv.add_module("torch_global_pool", torch.nn.AvgPool2d(kernel_size=5))
        
        self.csf = torch.nn.Sequential()
        self.csf.add_module("torch_fc1", torch.nn.Linear(256, 4096))
        self.csf.add_module("torch_relu3", torch.nn.ReLU())
        self.csf.add_module("torch_dropout1", torch.nn.Dropout(0.5))
        self.csf.add_module("torch_fc2", torch.nn.Linear(4096, 43))
        
        # Initialize conv layers and fc layers
        torch_init.normal(self.conv.state_dict()["torch_conv1.weight"], mean=0, std=0.01)
        torch_init.constant(self.conv.state_dict()["torch_conv1.bias"], 0.0)
        torch_init.normal(self.conv.state_dict()["torch_conv2.weight"], mean=0, std=0.01)
        torch_init.constant(self.conv.state_dict()["torch_conv2.bias"], 0.0)
        torch_init.normal(self.csf.state_dict()["torch_fc1.weight"], mean=0, std=0.01)
        torch_init.constant(self.csf.state_dict()["torch_fc1.bias"], 0.0)
        torch_init.normal(self.csf.state_dict()["torch_fc2.weight"], mean=0, std=0.01)
        torch_init.constant(self.csf.state_dict()["torch_fc2.bias"], 0.0)

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 256)
        return self.csf.forward(x)

torch_train_x = torch.stack([torch.Tensor(i.swapaxes(0,2).astype("float32")/255) for i in x_train])
torch_train_y = torch.LongTensor(y_train)
torch_valid_x = torch.stack([torch.Tensor(i.swapaxes(0,2).astype("float32")/255) for i in x_valid])
torch_valid_y = torch.LongTensor(y_valid)
torch_test_x = torch.stack([torch.Tensor(i.swapaxes(0,2).astype("float32")/255) for i in testImages])
torch_test_y = torch.LongTensor(testLabels)

torch_tensor_train_set = utils.TensorDataset(torch_train_x, torch_train_y)
torch_train_set = utils.DataLoader(torch_tensor_train_set, batch_size=batch_size, shuffle=True)
torch_tensor_valid_set = utils.TensorDataset(torch_valid_x, torch_valid_y)
torch_valid_set = utils.DataLoader(torch_tensor_valid_set, batch_size=batch_size, shuffle=True)
torch_tensor_test_set = utils.TensorDataset(torch_test_x, torch_test_y)
torch_test_set = utils.DataLoader(torch_tensor_test_set, batch_size=batch_size, shuffle=True)

torch_model_cpu = ConvNet()
torch_model_gpu = ConvNet().cuda()
max_total_batch = (len(x_train) / batch_size + 1) * epoch_num

def train(torch_model, optimizer, train_set, f, batch_count, gpu = False, epoch = None):
    if gpu:
        torch_model.cuda()
    torch_model.train() # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_set):
        batch_count += 1
        if gpu:
            data, target = data.cuda(), target.cuda()
        start = default_timer()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = torch_model(data)
        cost = torch.nn.CrossEntropyLoss(size_average=True)
        train_loss = cost(output, target)
        train_loss.backward()
        optimizer.step()

        # Save batch time
        train_batch_time = default_timer() - start
        f['.']['time']['train_batch'][batch_count-1] = train_batch_time

        # Get the accuracy of this batch
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = 100. * correct / len(data)

        # Save training loss and accuracy
        f['.']['cost']['train'][batch_count-1] = np.float32(train_loss.data[0])
        f['.']['accuracy']['train'][batch_count-1] = np.float32(acc)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},\tAccuracy: {}/{} ({:.0f}%)'.format(
                epoch, batch_idx * len(data), len(train_set.dataset),\
                100. * batch_idx / len(train_set), train_loss.data[0],\
                correct, len(data), acc))

    # Save batch marker
    f['.']['time_markers']['minibatch'][epoch] = np.float32(batch_count)

    return batch_count

def valid(torch_model, optimizer, valid_set, f, gpu = False, epoch = None):
    torch_model.eval() # Set the model to testing mode
    valid_loss = 0
    correct = 0
    for data, target in valid_set:
        if gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = torch_model(data)
        cost = torch.nn.CrossEntropyLoss(size_average=False)
        valid_loss += cost(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_loss /= len(valid_set.dataset)

    epoch_str = ""
    if epoch is not None:
        # Save validation loss and accuracy
        f['.']['cost']['loss'][epoch] = np.float32(valid_loss)
        f['.']['accuracy']['valid'][epoch] = np.float32(100. * correct / len(valid_set.dataset))
        epoch_str = "\nValid Epoch: {} ".format(epoch)
    else:
        # Save inference accuracy
        f['.']['infer_acc']['accuracy'][0] = np.float32(100. * correct / len(valid_set.dataset))
        epoch_str = "Test set: "
    print(epoch_str + 'Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_set.dataset),
        100. * correct / len(valid_set.dataset)))

# CPU & GPU
backends = ['gpu', 'cpu']
for b in backends:
    print("Run on {}".format(b))
    use_gpu = (b == 'gpu')
    batch_count = 0
    torch_model = torch_model_gpu if use_gpu else torch_model_cpu
    optimizer = optim.SGD(torch_model.parameters(), lr=0.01, momentum=0.9)

    filename = root + "/saved_data/callback_data_pytorch_{}.h5".format(b)
    f = DLHelper.init_h5py(filename, epoch_num, max_total_batch)
    try:
        for epoch in range(epoch_num):

            # Start training and save start and end time
            f['.']['time']['train']['start_time'][0] = time.time()
            batch_count = train(torch_model, optimizer, torch_train_set, f, batch_count, use_gpu, epoch)
            f['.']['time']['train']['end_time'][0] = time.time()

            # Validation per epoch
            valid(torch_model, optimizer, torch_valid_set, f, use_gpu, epoch)

        # Save total batch count
        f['.']['config'].attrs["total_minibatches"] = batch_count
        f['.']['time_markers'].attrs['minibatches_complete'] = batch_count

        # Final test
        valid(torch_model, optimizer, torch_test_set, f, use_gpu)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
    finally:
        print("Close file descriptor")
        f.close()