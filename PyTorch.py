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

torch_model = ConvNet()
optimizer = optim.SGD(torch_model.parameters(), lr=0.01, momentum=0.9)

def train(epoch):
    torch_model.train()
    for batch_idx, (data, target) in enumerate(torch_train_set):
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = torch_model(data)
        cost = torch.nn.CrossEntropyLoss(size_average=True)
        loss = cost(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(torch_train_set.dataset),
                100. * batch_idx / len(torch_train_set), loss.data[0]))
def test():
    torch_model.eval()
    test_loss = 0
    correct = 0
    for data, target in torch_test_set:
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = torch_model(data)
        cost = torch.nn.CrossEntropyLoss(size_average=False)
        test_loss += cost(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(torch_test_set.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(torch_test_set.dataset),
        100. * correct / len(torch_test_set.dataset)))

for epoch in range(1, epoch_num + 1):
    train(epoch)
test()
