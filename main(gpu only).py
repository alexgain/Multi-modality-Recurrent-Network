import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data

import struct
from copy import deepcopy, copy
from time import time, sleep
import gc

from sklearn.preprocessing import normalize
##import warnings
##warnings.filterwarnings("ignore")

np.random.seed(1)

##torch.set_default_tensor_type('torch.cuda.FloatTensor')

###                    ###
### Data preprocessing ###
###                    ###

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

xtest = read_idx('t10k-images.idx3-ubyte')
ytest = read_idx('t10k-labels.idx1-ubyte')
xtrain = read_idx('train-images.idx3-ubyte')
ytrain = read_idx('train-labels.idx1-ubyte')

##when using softmax:
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
ytrain = to_categorical(ytrain, 10)
ytest = to_categorical(ytest, 10)

##reshaping data:
xtrain = np.reshape(xtrain,(xtrain.shape[0],784))
xtest = np.reshape(xtest,(xtest.shape[0],784))
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain /= 255
xtest /= 255

##instantiation into pytorch tensors and data loading:
xtrain = torch.from_numpy(xtrain)
ytrain = torch.from_numpy(ytrain)
train = torch.utils.data.TensorDataset(xtrain, ytrain)
xtest = torch.from_numpy(xtest)
ytest = torch.from_numpy(ytest)
test = torch.utils.data.TensorDataset(xtest, ytest)

###                        ###
### Some misc. function(s) ###
###                        ###

class RangeNormalize(object):
    """
    Given min_val: (R, G, B) and max_val: (R,G,B),
    will normalize each channel of the th.*Tensor to
    the provided min and max values.
    Works by calculating :
        a = (max'-min')/(max-min)
        b = max' - a * max
        new_value = a * value + b
    where min' & max' are given values, 
    and min & max are observed min/max for each channel
    
    Arguments
    ---------
    min_range : float or integer
        Min value to which tensors will be normalized
    max_range : float or integer
        Max value to which tensors will be normalized
    fixed_min : float or integer
        Give this value if every sample has the same min (max) and 
        you know for sure what it is. For instance, if you
        have an image then you know the min value will be 0 and the
        max value will be 255. Otherwise, the min/max value will be
        calculated for each individual sample and this will decrease
        speed. Dont use this if each sample has a different min/max.
    fixed_max :float or integer
        See above
    Example:
        >>> x = th.rand(3,5,5)
        >>> rn = RangeNormalize((0,0,10),(1,1,11))
        >>> x_norm = rn(x)
    Also works with just one value for min/max:
        >>> x = th.rand(3,5,5)
        >>> rn = RangeNormalize(0,1)
        >>> x_norm = rn(x)
    """
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

###                      ###
### Define torch network ###
###                      ###

class Net(nn.Module):
    def __init__(self, adj_net, adj_in, W_net, W_in, dna, W_dna):
        super(Net, self).__init__()

        #basic building block inits:
##        self.adj_net = adj_net
        self.adj_in = adj_in
        self.W_net = W_net
        self.W_in = W_in
        self.W_dna = W_dna
        self.dna = dna
        self.state = Variable(torch.Tensor(adj_net.shape[0],1).uniform_(0, 0.1).cuda())
        self.prev_state = copy(self.state)
        
        #dense out layer needed for classification:
##        self.ff_dna_net = nn.Linear(500, self.adj_net.shape[0]*self.adj_net.shape[1], bias = True).cuda()
        self.ff_out = nn.Linear(self.state.shape[0], 10, bias = True) #output     
        
        #activations:
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()

        #re-computing adj_net:
##        self.adj_net = torch.round(normalize_01(self.relu(dna)))
##        self.adj_net = self.adj_net.resize(self.W_net.shape[0],self.W_net.shape[1])
##        self.adj_net = torch.round(normalize_01(self.relu(self.W_dna.mm(self.dna))))
        
    def update(self, input_data):

        #main update equation and computation:
##        new_state = self.relu((self.adj_net*self.W_net).mm(self.prev_state) + (self.adj_in*self.W_in).mm(torch.transpose(input_data,0,1)))
        new_state = self.relu((torch.round(normalize_01(self.relu(self.W_dna.mm(self.dna))))*self.W_net).mm(self.prev_state) + (self.adj_in*self.W_in).mm(torch.transpose(input_data,0,1)))

        #updating variables:
        self.prev_state = copy(self.state)
        self.state = new_state

        return new_state

    def forward(self, input_data):

        if input_data.shape[0] != self.state.shape[0]:
            self.state = Variable(torch.Tensor(adj_net.shape[0],input_data.shape[0]).uniform_(0, 0.1).cuda())
            self.prev_state = copy(self.state)
    
        for k in range(3):
            self.update(input_data)

        return self.sm(self.ff_out(torch.transpose(self.state,0,1)))
        
    def save_weights(self, file_name):
        params = list(self.parameters())
        L = len(params)-2
        params = params[:L]
        Ws = [w.cuda().data.numpy().T for w in params]
        np.savez(file_name, args = [w for w in Ws])

###hyper-parameters:
input_size = 28*28
net_size = 500
num_classes = 10

###initializing adjacency matrices and weights:

##network adjacency and weights:
adj_net = torch.round(torch.Tensor(net_size, net_size).uniform_(0, 1))
#adj_net = adj_net + adj_net.transpose(0,1) - adj_net.diag()
W_net = torch.Tensor(net_size, net_size).normal_(0, 0.01)

##input adjacency and weights:
adj_in = torch.round(torch.Tensor(net_size, input_size).uniform_(0, 1))
#adj_in = adj_in + adj_in.transpose(0,1) - adj_in.diag()
W_in = torch.Tensor(net_size, input_size).normal_(0, 0.01)

##changing tensors to Variables:
adj_net, W_net, adj_in, W_in = Variable(adj_net.cuda(), requires_grad = False), Variable(W_net.cuda(), requires_grad = True), Variable(adj_in.cuda(), requires_grad = False), Variable(W_in.cuda(), requires_grad = True)

###initializing W_dna and dna:

#computing via random uniform:
W_dna = torch.Tensor(500,500).uniform_(0, 1)
dna = torch.Tensor(500,500).uniform_(0, 1)

#changing to dna stuff to Variables:
W_dna = Variable(W_dna.cuda(), requires_grad = True)
dna = Variable(dna.cuda(), requires_grad = False)

#computing adj_net:
def relu_tensor(x):
    return x*((x>0).float())

##adj_net = Variable(torch.round(normalize_01(relu_tensor(W_dna.mm(dna)))).cuda(), requires_grad = False)
##adj_net = torch.round(normalize_01(nn.ReLU()(W_dna.mm(dna)))).cuda()

###defining network:
my_net = Net(adj_net, adj_in, W_net, W_in, dna, W_dna)
my_net.cuda()

###test:
##inputyo = Variable(torch.Tensor(784,1).uniform_(0,1))
##out = my_net.update(inputyo)

###                       ###
### Loss and optimization ###
###                       ###

LR = 0.01
loss_metric = nn.MSELoss()
optimizer = torch.optim.Adam(my_net.parameters(), lr = LR)

###          ###
### Training ###
###          ###

#Some more hyper-params and initializations:
epochs = 30
N = xtrain.shape[0]
BS = 32

train_loader = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=BS, shuffle=False)

##printing train statistics:
# Test the Model
correct = 0
total = 0
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    images = Variable(images.view(-1, 28*28))
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
    images, labels = images.cuda(), labels.cuda()
    images = Variable(images.view(-1, 28*28))
    outputs = my_net(images)
    _, predicted = torch.max(outputs.data, 1)
    labels = torch.max(labels.float(),1)[1]
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

        ##cuda shit:
        x = x.cuda()
        
        ##data preprocessing for optimization purposes:
        x = Variable(x)
        y = Variable(y.float().cuda()) #MSE 1-d output version

        ###regular BP gradient update:
##        print(my_net.W_dna)
        
        optimizer.zero_grad()
        outputs = my_net.forward(x)
        loss = loss_metric(outputs,y)
        loss.backward()

##        print(my_net.W_dna)
                
        ##performing update:
        optimizer.step()
        
        ##printing statistics:
        if (i+1) % 1875 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, epochs, i+1, N//BS, loss.data[0]))

            ##printing train statistics:
            # Test the Model
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.cuda(), labels.cuda()
                images = Variable(images.view(-1, 28*28))
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
                images, labels = images.cuda(), labels.cuda()
                images = Variable(images.view(-1, 28*28))
                outputs = my_net(images)
                _, predicted = torch.max(outputs.data, 1)
                labels = torch.max(labels.float(),1)[1]
            ##    predicted = torch.round(outputs.data).view(-1).long()
                total += labels.size(0)
                correct += (predicted.float() == labels.float()).sum()

            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


    ##time-keeping 2:
    time2 = time()
    print('Elapsed time for epoch:',time2 - time1,'s')
    print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')
    print()

t2 = time()
print((t2 - t1)/60,'total minutes elapsed')
             
        

####printing train statistics:
### Test the Model
##correct = 0
##total = 0
##for images, labels in train_loader:
##    images, labels = images.cuda(), labels.cuda()
##    images = Variable(images.view(-1, 28*28))
##    outputs = my_net(images)
##    _, predicted = torch.max(outputs.data, 1)
##    labels = torch.max(labels.float(),1)[1]
####    predicted = torch.round(outputs.data).view(-1).long()
##    total += labels.size(0)
##    correct += (predicted.float() == labels.float()).sum()
##
##print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))
##
####printing test statistics:
### Test the Model
##correct = 0
##total = 0
##for images, labels in test_loader:
##    images, labels = images.cuda(), labels.cuda()
##    images = Variable(images.view(-1, 28*28))
##    outputs = my_net(images)
##    _, predicted = torch.max(outputs.data, 1)
##    labels = torch.max(labels.float(),1)[1]
####    predicted = torch.round(outputs.data).view(-1).long()
##    total += labels.size(0)
##    correct += (predicted.float() == labels.float()).sum()
##
##print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
##



