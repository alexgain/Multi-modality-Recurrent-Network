import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import struct
from copy import deepcopy, copy
from time import time, sleep
import gc

from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

##import warnings
##warnings.filterwarnings("ignore")

np.random.seed(1)
cuda_boole = 1

##torch.set_default_tensor_type('torch.cuda.FloatTensor')

###                    ###
### Data preprocessing ###
###                    ###

##xtrain, ytrain = load_mnist('data/fashion', kind='train')
##xtest, ytest = load_mnist('data/fashion', kind='t10k')

####when using softmax:
##def to_categorical(y, num_classes):
##    """ 1-hot encodes a tensor """
##    return np.eye(num_classes, dtype='uint8')[y]
##ytrain = to_categorical(ytrain, 10)
##ytest = to_categorical(ytest, 10)
##
####reshaping data:
##xtrain = np.reshape(xtrain,(xtrain.shape[0],1,28,28))
##xtest = np.reshape(xtest,(xtest.shape[0],1,28,28))
##xtrain = xtrain.astype('float32')
##xtest = xtest.astype('float32')
##xtrain /= 255
##xtest /= 255

##instantiation into pytorch tensors and data loading:
##xtrain = torch.from_numpy(xtrain)
##ytrain = torch.from_numpy(ytrain)
##train = torch.utils.data.TensorDataset(xtrain, ytrain)
##xtest = torch.from_numpy(xtest)
##ytest = torch.from_numpy(ytest)
##test = torch.utils.data.TensorDataset(xtest, ytest)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)

###                        ###
### Some misc. function(s) ###
###                        ###

class RangeNormalize(object):
    def __init__(self, 
                 min_val, 
                 max_val):
        """
        Normalize a tensor between a min and max value
        Arguments
        ---------
        min_val : float
            lower bound of normalized tensor
        max_val : float
            upper bound of normalized tensor
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _min_val = _input.min()
            _max_val = _input.max()
            a = (self.max_val - self.min_val) / (_max_val - _min_val)
            b = self.max_val- a * _max_val
            _input = _input.mul(a).add(b)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]

normalize_01 = RangeNormalize(0,1) #needed for torch tensor normalization between 0 and 1

###                                    ###
### Define torch preprocessing network ###
###                                    ###

class ff_Net(nn.Module):
    def __init__(self):
        super(ff_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print(x.shape)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return x

###                         ###
### Define main torch model ###
###                         ###

##some hyper-parameters:
input_size = 28*28*3
net_size = 1000
num_classes = 10


class Net(nn.Module):
    def __init__(self, adj_net = None, adj_in = None, W_net = None, W_in = None, dna = None, W_dna = None, ff_net = None):
        super(Net, self).__init__()

        ##feedforward part:
        out_shape = 50
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(25*20, 50)
##        self.fc2 = nn.Linear(50, 10)

        ###initializing adjacency matrices and weights:

        ##network adjacency and weights:
        self.adj_net = torch.round(torch.Tensor(net_size, net_size).uniform_(0, 1))
        self.W_net = torch.Tensor(net_size, net_size).normal_(0, 0.01)

        ##input adjacency and weights:
##        self.adj_in = torch.round(torch.Tensor(net_size, input_size).uniform_(0, 1))
##        self.W_in = torch.Tensor(net_size, input_size).normal_(0, 0.01)
        self.adj_in = torch.ones(net_size, input_size)
        self.W_in = torch.ones(net_size, input_size)

        ##changing tensors to nn.Parameters/Variables:
        if cuda_boole:
        ##    adj_net, W_net, adj_in, W_in = Variable(adj_net.cuda(), requires_grad = False), nn.Parameter(W_net.cuda(), requires_grad = True), Variable(adj_in.cuda(), requires_grad = False), nn.Parameter(W_in.cuda(), requires_grad = True)
            self.adj_net, self.W_net, self.adj_in, self.W_in = Variable(self.adj_net.cuda(), requires_grad = True), nn.Parameter(self.W_net.cuda(), requires_grad = True), Variable(self.adj_in.cuda(), requires_grad = True), nn.Parameter(self.W_in.cuda(), requires_grad = True)
        else:
        ##    adj_net, W_net, adj_in, W_in = Variable(adj_net, requires_grad = False), nn.Parameter(W_net, requires_grad = True), Variable(adj_in, requires_grad = False), nn.Parameter(W_in, requires_grad = True)
            self.adj_net, self.W_net, self.adj_in, self.W_in = Variable(self.adj_net, requires_grad = True), nn.Parameter(self.W_net, requires_grad = True), Variable(self.adj_in, requires_grad = True), nn.Parameter(self.W_in, requires_grad = True)

        ###initializing W_dna and dna:

        #computing via random uniform:
        self.W_dna = torch.Tensor(net_size,net_size).uniform_(0, 1)
        self.dna = torch.Tensor(net_size,net_size).uniform_(0, 1)

        #changing to dna stuff to nn.Parameters/Variables:
        if cuda_boole:
            self.W_dna = nn.Parameter(self.W_dna.cuda(), requires_grad = True)
            self.dna = Variable(self.dna.cuda(), requires_grad = False)
        else:
            self.W_dna = nn.Parameter(self.W_dna, requires_grad = True)
            self.dna = Variable(self.dna, requires_grad = False)

        seed = torch.Tensor(self.adj_net.shape[0]).uniform_(0, 0.1)
        
        if cuda_boole:
            self.state = Variable(seed.cuda(), requires_grad = True)
        else:
            self.state = Variable(seed, requires_grad = True)
                    
        #dense out layer needed for classification:
        self.ff_out = nn.Linear(self.state.shape[0], 10, bias = False) #output     
        
        #activations:
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()

        ##Manually setting initialization:
        if adj_net is not None:
            self.adj_net = adj_net
        if adj_in is not None:
            self.adj_in = adj_in
        if W_net is not None:
            self.W_net = W_net
        if W_in is not None:
            self.W_in = W_in
        if W_dna is not None:
            self.W_dna = W_dna
        if dna is not None:
            self.dna = dna
        if ff_net is not None:
            self.ff_net = ff_net
            out_shape = list(self.ff_net.modules())[-1]
            out_shape = out_shape.out_features
            self.adj_in, self.W_in = None, None
            self.adj_in = nn.Parameter(torch.round(torch.Tensor(net_size, out_shape).uniform_(0, 1)))
            self.W_in = nn.Parameter(torch.Tensor(net_size, out_shape).normal_(0, 0.01))
            
##        else:
##            self.ff_net = None
##        out_shape = list(self.ff_net.modules())[-1]
##        out_shape = out_shape.out_features
        self.adj_in, self.W_in = None, None
        self.adj_in = nn.Parameter(torch.round(torch.Tensor(net_size, out_shape).uniform_(0, 1)))
        self.W_in = nn.Parameter(torch.Tensor(net_size, out_shape).normal_(0, 0.01))


    def ff_net_forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 25*20)
        x = F.relu(self.fc1(x))
        
##        x = F.dropout(x, training=self.training)
##        x = F.relu(self.fc2(x))
        return x
        
        
    def forward(self, input_data):

        ##Doing pre-processing if ff_net is available:
##        if self.ff_net is not None:
##            input_data = self.ff_net(input_data)

        input_data = self.ff_net_forward(input_data)

        ##Setting initial state for each datapoint:
        seed = torch.Tensor(input_data.shape[0],self.adj_net.shape[0]).uniform_(0, 0.1)

        if cuda_boole:
            self.state = Variable(seed.cuda(), requires_grad = True)
        else:
            self.state = Variable(seed, requires_grad = True)
        
        ##Updating adj_net (and adj_in?):
        self.adj_net = normalize_01(self.relu(self.W_dna.mm(self.dna)))
        self.adj_net = (1 / (1+torch.exp(-(80*(self.adj_net - 0.5)))))

        ##Forward propagation:
        x_bias = ((self.adj_in*self.W_in).matmul(torch.transpose(input_data,0,1)))

        x = self.relu((self.adj_net*self.W_net).matmul(self.state.t())) + x_bias
        x = self.relu((self.adj_net*self.W_net).matmul(x)) + x_bias
        x = self.relu((self.adj_net*self.W_net).matmul(x)) + x_bias
                
        return self.sm(self.ff_out(x.t()))
        
    def plot_adj(self):
        self.adj_net = normalize_01(self.relu(self.W_dna.mm(self.dna)))
        self.adj_net = (1 / (1+torch.exp(-(80*(self.adj_net - 0.5)))))

        plt.imshow(self.adj_net.cpu().data.numpy())
        plt.show()

    def save_weights(self, file_name):
        params = list(self.parameters())
        L = len(params)-2
        params = params[:L]
        if cuda_boole:
            Ws = [w.cuda().data.numpy().T for w in params]
        else:
            Ws = [w.numpy().T for w in params]
        np.savez(file_name, args = [w for w in Ws])

###defining network:
if cuda_boole:
##    ff_net = ff_Net().cuda()
##    my_net = Net(ff_net = ff_net).cuda()
    my_net = Net().cuda()
else:
##    ff_net = ff_Net()
##    my_net = Net(ff_net = ff_net)
    my_net = Net()

###                       ###
### Loss and optimization ###
###                       ###

LR = 0.0001
loss_metric = nn.MSELoss()
optimizer = torch.optim.Adam(my_net.parameters(), lr = LR)

###          ###
### Training ###
###          ###

#Some more hyper-params and initializations:
epochs = 60
##N = xtrain.shape[0]
N = 50000
BS = 200

##train_loader = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True)
##test_loader = torch.utils.data.DataLoader(test, batch_size=BS, shuffle=False)

##printing train statistics:
# Test the Model
correct = 0
total = 0
for images, labels in train_loader:
    if cuda_boole:
        images, labels = images.cuda(), labels.cuda()
    images = Variable(images)#.view(-1, 28*28))
    outputs = my_net.forward(images)
    _, predicted = torch.max(outputs.data, 1)
##    labels = torch.max(labels.float(),1)[1]
##    predicted = torch.round(outputs.data).view(-1).long()
    total += labels.size(0)
    correct += (predicted.float() == labels.float()).sum()

print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))

##printing test statistics:
# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    if cuda_boole:
        images, labels = images.cuda(), labels.cuda()
    images = Variable(images)#.view(-1, 28*28))
    outputs = my_net(images)
    _, predicted = torch.max(outputs.data, 1)
##    labels = torch.max(labels.float(),1)[1]
##    predicted = torch.round(outputs.data).view(-1).long()
    total += labels.size(0)
    correct += (predicted.float() == labels.float()).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

###training loop:
t1 = time()
for epoch in range(epochs):

    ##time-keeping 1:
    time1 = time()

    for i, (x,y) in enumerate(train_loader):

        ##cuda:
        if cuda_boole:
            x = x.cuda()
            y = y.float().cuda()
        else:
            y = y.float()
        
        ##data preprocessing for optimization purposes:
        x = Variable(x)
        y = Variable(y) #MSE 1-d output version

        ###regular BP gradient update:
##        print(my_net.W_dna)
        
        optimizer.zero_grad()
        outputs = my_net.forward(x)
        loss = loss_metric(outputs,y)
        loss.backward(retain_graph = True)

##        my_net.W_dna.grad = my_net.W_dna.grad*(1e10)
##        my_net.W_in.grad = 0*my_net.W_in.grad
##        my_net.W_net.grad = 0*my_net.W_net.grad
##        my_net.ff_out.weight.grad = 0*my_net.ff_out.weight.grad
##        my_net.ff_out.weight.bias = 0*my_net.ff_out.weight.bias
        

##        print(my_net.W_dna)
                
        ##performing update:
        optimizer.step()
        
        ##printing statistics:
        if (i+1) % np.floor(N/BS) == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, epochs, i+1, N//BS, loss.data[0]))

            ##printing train statistics:
            # Test the Model
            correct = 0
            total = 0
            for images, labels in train_loader:
                if cuda_boole:
                    images, labels = images.cuda(), labels.cuda()
                images = Variable(images)#.view(-1, 28*28))
                outputs = my_net(images)
                _, predicted = torch.max(outputs.data, 1)
                labels = torch.max(labels.float(),1)[1]
            ##    predicted = torch.round(outputs.data).view(-1).long()
                total += labels.size(0)
                correct += (predicted.float() == labels.float()).sum()

            print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))

            ##printing test statistics:
            # Test the Model
            correct = 0
            total = 0
            for images, labels in test_loader:
                if cuda_boole:
                    images, labels = images.cuda(), labels.cuda()
                images = Variable(images)#.view(-1, 28*28))
                outputs = my_net(images)
                _, predicted = torch.max(outputs.data, 1)
                labels = torch.max(labels.float(),1)[1]
            ##    predicted = torch.round(outputs.data).view(-1).long()
                total += labels.size(0)
                correct += (predicted.float() == labels.float()).sum()

            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

            yes_adj = (my_net.adj_net>=0.9).float().sum().cpu().data.numpy()[0] / net_size**2
            no_adj = (my_net.adj_net<=0.1).float().sum().cpu().data.numpy()[0] / net_size**2
            print('% greater than 0.9:',yes_adj)
            print('% less than 0.1:',no_adj)
            print('Sum:',yes_adj + no_adj)



    ##time-keeping 2:
    time2 = time()
    print('Elapsed time for epoch:',time2 - time1,'s')
    print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')
    print()

t2 = time()
print((t2 - t1)/60,'total minutes elapsed')

    
