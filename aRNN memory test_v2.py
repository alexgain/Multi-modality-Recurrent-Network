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

cuda_boole = torch.cuda.is_available()

##torch.set_default_tensor_type('torch.cuda.FloatTensor')

###                    ###
### Data preprocessing ###
###                    ###

input_num = 5000
dim_num = 9
seq_num = 20

train = np.random.randint(2, size=(input_num,dim_num,seq_num))
test = np.random.randint(2, size=(int(input_num/10),dim_num,seq_num))

xtrain = np.insert(train, train.shape[1], 0, axis=1)
xtrain = np.insert(xtrain, train.shape[2], 0, axis=2)
xtrain[:,-1,-1] += 1
ytrain, xtrain = np.concatenate((xtrain * 0, xtrain), axis = 2), np.concatenate((xtrain,xtrain * 0), axis = 2)

xtrain = xtrain[:,:,:xtrain.shape[2]-1]
ytrain = ytrain[:,:,:ytrain.shape[2]-1]

xtest = np.insert(test, test.shape[1], 0, axis=1)
xtest = np.insert(xtest, test.shape[2], 0, axis=2)
xtest[:,-1,-1] += 1
ytest, xtest = np.concatenate((xtest * 0, xtest), axis = 2), np.concatenate((xtest,xtest * 0), axis = 2)

xtest = xtest[:,:,:xtest.shape[2]-1]
ytest = ytest[:,:,:ytest.shape[2]-1]

xtrain = torch.Tensor(xtrain)
xtest = torch.Tensor(xtest)
ytrain = torch.Tensor(ytrain).contiguous()
ytest = torch.Tensor(ytest).contiguous()

train = torch.utils.data.TensorDataset(xtrain, ytrain)
test = torch.utils.data.TensorDataset(xtest, ytest)

##for L in range(1):
##    input_num2 = 5
##    dim_num2 = 9
##    seq_num2 = 20
##
##    train2 = np.random.randint(2, size=(input_num2,dim_num2,seq_num2))
####    test = np.random.randint(2, size=(int(input_num/10),dim_num,seq_num))
##
##    xtrain2 = np.insert(train2, train2.shape[1], 0, axis=1)
##    xtrain2 = np.insert(xtrain2, train2.shape[2], 0, axis=2)
##    xtrain2[:,-1,-1] += 1
##    ytrain2, xtrain2 = np.concatenate((xtrain2 * 0, xtrain2), axis = 2), np.concatenate((xtrain2,xtrain2 * 0), axis = 2)
##
##    xtrain2 = xtrain2[:,:,:xtrain2.shape[2]-1]
##    ytrain2 = ytrain2[:,:,:ytrain2.shape[2]-1]
##
##    xtrain2 = Variable(torch.Tensor(xtrain2).cuda())
##    outs2 = my_net.forward_seq(xtrain2)
    
##if cuda_boole:
##    xtrain, xtest, ytrain, ytest = xtrain.cuda(), xtest.cuda(), ytrain.cuda(), ytest.cuda()


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

###                         ###
### Define main torch model ###
###                         ###

##some hyper-parameters:
#input_num = train.shape[0]
#dim_num = train.shape[1]
#seq_num = train.shape[2]
net_size = 500
num_classes = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ##normalize and torch round sub-functions:
        def round_diff(x):
            return (1 / (1+torch.exp(-(100*(x - 0.5)))))
        self.round_diff = round_diff

        ##feedforward preprocessing:
        out_shape = 80
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 96, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(25*96, out_shape)
        self.fc2 = nn.Linear(50, 10)

        ff_preprocessing = nn.Sequential(self.conv1,nn.ReLU(),self.conv2,self.conv2_drop,nn.ReLU(),self.fc1,nn.ReLU(),self.fc2,nn.ReLU())

        ##feedforwardfor adj_in

        #MLP:
        out_shape = (net_size, dim_num+1)
        self.ff_in1 = nn.Linear(dim_num+1,300)
        self.ff_in2 = nn.Linear(300,300)
##        self.ff_in3 = nn.Linear(300,(dim_num+1)*net_size)
        self.ff_in3 = nn.Conv1d(1,1,300,(dim_num+1)*net_size)

        ff_in_seq = nn.Sequential(self.ff_in1,nn.ReLU(),self.ff_in2,nn.ReLU(),self.ff_in3,nn.ReLU())

        #CNN:
##        out_shape = (net_size, dim_num+1)
##        self.conv_in1 = nn.Linear(dim_num+1,net_size)
##        self.conv_in2 = nn.Linear(net_size,net_size)
##        self.conv_in3 = nn.Linear(net_size,net_size**2)

        def adj_in_calc(input_data):
            N = input_data.shape[0]
            adj = ff_in_seq(input_data)
            adj = adj.view(N,net_size,dim_num+1)
            adj = normalize_01(adj)
            adj = round_diff(adj)
            return adj

        self.adj_in_calc = adj_in_calc

        ##feedforward for adj_net:

        #MLP:
        out_shape = (net_size, net_size)
        self.ff_net1 = nn.Linear(dim_num+1,net_size)
        self.ff_net2 = nn.Linear(net_size,net_size)
        self.ff_net3 = nn.Linear(net_size,net_size**2)

        ff_net_seq = nn.Sequential(self.ff_net1,nn.ReLU(),self.ff_net2,nn.ReLU(),self.ff_net3,nn.ReLU())

        #CNN:
##        out_shape = (net_size, dim_num+1)
##        self.conv_in1 = nn.Linear(dim_num+1,net_size)
##        self.conv_in2 = nn.Linear(net_size,net_size)
##        self.conv_in3 = nn.Linear(net_size,net_size**2)

        def adj_net_calc(input_data):
            N = input_data.shape[0]
            adj = ff_net_seq(input_data)
            adj = adj.view(N,net_size,net_size)
            adj = normalize_01(adj)
            adj = round_diff(adj)
            return adj

        self.adj_net_calc = adj_net_calc
        
        ##initializing state:

        seed = torch.Tensor(net_size).uniform_(0, 0.1)
        
        if cuda_boole:
            self.state = Variable(seed.cuda(), requires_grad = True)
        else:
            self.state = Variable(seed, requires_grad = True)

        ##initializing W_in and W_net:

        self.W_net = torch.Tensor(net_size,net_size).uniform_(0, 1)
        self.W_in = torch.Tensor(net_size,dim_num+1).uniform_(0, 1)
        if cuda_boole:
            self.W_net = nn.Parameter(self.W_net.cuda(), requires_grad = True)
            self.W_in = nn.Parameter(self.W_in.cuda(), requires_grad = True)
        else:
            self.W_net = nn.Parameter(self.W_net, requires_grad = True)
            self.W_in = nn.Parameter(self.W_in, requires_grad = True)
                    
        #dense out layer needed for classification:
        self.ff_out = nn.Linear(self.state.shape[0], dim_num+1)     
        
        #activations:
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()        
        
    def forward_seq(self, input_data):

        ##Setting initial state:
        seed = torch.Tensor(input_data.shape[0],net_size).uniform_(0, 0.1)

        if cuda_boole:
            self.state = Variable(seed.cuda(), requires_grad = True)
        else:
            self.state = Variable(seed, requires_grad = True)

        ##Getting adjacencies:
        adj_ins = self.adj_in_calc(input_data[:,:,0])
        adj_nets = self.adj_net_calc(input_data[:,:,0])

        ##Forward propagation:
        outs = []
        x_bias = ((adj_ins*self.W_in).matmul(input_data[:,:,0].unsqueeze(2))).squeeze()
        x = self.relu(((adj_nets*self.W_net).matmul(self.state.unsqueeze(2))).squeeze() + x_bias)
        x = self.relu(((adj_nets*self.W_net).matmul(x.unsqueeze(2))).squeeze() + x_bias)
        outs.append(self.ff_out(x))
##        outs.append(normalize_01(self.ff_out(x.t())))
##        outs.append(F.log_softmax(self.ff_out(x.t()),dim=1))
        for t in range(1,input_data.shape[2]):

            ##Getting adjacencies:
            adj_ins = self.adj_in_calc(input_data[:,:,t])
            adj_nets = self.adj_net_calc(input_data[:,:,t])

            ##Forward propagation:            
            x_bias = ((adj_ins*self.W_in).matmul(input_data[:,:,t].unsqueeze(2))).squeeze()
            x = self.relu(((adj_nets*self.W_net).matmul(self.state.unsqueeze(2))).squeeze() + x_bias)
            x = self.relu(((adj_nets*self.W_net).matmul(x.unsqueeze(2))).squeeze() + x_bias)
            outs.append(self.ff_out(x))

##            outs.append(normalize_01(self.ff_out(x.t())))
##            outs.append(F.log_softmax(self.ff_out(x.t()),dim=1))
        #outs  == list of (N x d) elements of length 2s + 1
        outs = torch.stack(outs,2)
        #outs = outs.permute(0,2,1)
        return outs
        
        
    def plot_adj(self):
        self.adj_net = normalize_01(self.relu(self.W_dna.mm(self.dna)))
        self.adj_net = (1 / (1+torch.exp(-(100*(self.adj_net - 0.5)))))

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

LR = 0.05
loss_metric = nn.MSELoss()
##loss_metric = nn.CrossEntropyLoss()
##loss_metric = nn.NLLLoss()
##optimizer = torch.optim.Adam(my_net.parameters(), lr = LR)
##optimizer = torch.optim.SGD(my_net.parameters(), lr = LR, momentum = 0.9)
optimizer = torch.optim.RMSprop(my_net.parameters(), lr = LR/1000, momentum = 0.9)

###          ###
### Training ###
###          ###

#Some more hyper-params and initializations:
epochs = 50000
BS = 32

train_loader = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=int(input_num/10), shuffle=False)

##preds = my_net.forward_seq(xtrain)
##acc = ((np.equal(np.round(preds.cpu().data.numpy()) ,ytrain.cpu().data.numpy()))*1).mean()
##
##test_preds = my_net.forward_seq(xtest)
##test_acc = ((np.equal(np.round(test_preds.cpu().data.numpy()), ytest.cpu().data.numpy()))*1).mean()

print('Epoch #:','Initialization')
#print('Accuracy:',acc*100, '%')
#print('Test Accuracy:',test_acc*100, '%')


###training loop:
t1 = time()
for epoch in range(epochs):

    ##time-keeping 1:
    time1 = time()

    for i, (x,y) in enumerate(train_loader):
        if cuda_boole:
            xtrain = x.cuda()
            ytrain = y.cuda()

        xtrain = Variable(xtrain)
        ytrain = Variable(ytrain)

        ##grad udpate:
        my_net.zero_grad()
        preds = my_net.forward_seq(xtrain)
        loss = loss_metric(preds,ytrain)
    ##    loss = ((preds - ytrain)**2).mean()
        loss.backward()
##        my_net.W_dna.grad *= 5
        optimizer.step()
        acc = ((np.equal(np.round(preds.cpu().data.numpy()), ytrain.cpu().data.numpy()))*1).mean()
        loss2 = loss.cpu().data.numpy()[0]
        del x,y,xtrain,ytrain,preds,loss

    for i, (x,y) in enumerate(test_loader):
        if cuda_boole:
            xtest = x.cuda()
            ytest = y.cuda()

        xtest = Variable(xtest)
        ytest = Variable(ytest)

        test_preds = my_net.forward_seq(xtest)
        test_acc = ((np.equal(np.round(test_preds.cpu().data.numpy()), ytest.cpu().data.numpy()))*1).mean()
        del x,y,xtest,ytest,test_preds
        
    print('Epoch #:',epoch)
    print('Train Accuracy:',acc*100, '%')
    print('Test Accuracy:',test_acc*100, '%')
    print('Current loss:',loss2)

    del acc, test_acc, loss
    
##    yes_adj = (my_net.adj_net>=0.9).float().sum().cpu().data.numpy()[0] / net_size**2
##    no_adj = (my_net.adj_net<=0.1).float().sum().cpu().data.numpy()[0] / net_size**2
##    print('% greater than 0.9:',yes_adj)
##    print('% less than 0.1:',no_adj)
##    print('Sum:',yes_adj + no_adj)

    ##time-keeping 2:
    time2 = time()
    print('Elapsed time for epoch:',time2 - time1,'s')
    print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')
    print()

t2 = time()
print((t2 - t1)/60,'total minutes elapsed')

    
