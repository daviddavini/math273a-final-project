import math
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import argparse

from net import ConvolutionalRegularizer, FullyConnectedNetwork
import datetime
import os

save_dir = "learn_data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=10, type=float, help='learning rate')
num_epochs = 100
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
best_acc = 0  # best test accuracy

# Data
print('==> Preparing data..')
# Generate random input/output vector pairs
NUM_DATA = 100
INPUT_SIZE = 100
HIDDEN_SIZE = 100
OUTPUT_SIZE = 100

def checkered_matrix(m, n):
    M = torch.zeros(m, n)
    for i in range(m):
        for j in range(n):
            M[i][j] = (i % 2) ^ (j % 2)
    return M

def sinusoid_matrix(m, n):
    M = torch.zeros(m, n)
    for i in range(m):
        M[i][int((n//2 - 1) * math.sin(2 * math.pi * i / n)) + n//2] = 1.0
    return M

def line_matrix(slope, m, n):
    M = torch.zeros(m, n)
    for i in range(m):
        M[i][int(slope*i) % n] = 1.0
    return M

X = torch.randn(NUM_DATA, INPUT_SIZE)
# X = torch.zeros(NUM_DATA, INPUT_SIZE)
# X = torch.eye(NUM_DATA, INPUT_SIZE)
# X = line_matrix(0.5, NUM_DATA, INPUT_SIZE)
# X.requires_grad_(True)

# Y = torch.randn(NUM_DATA, OUTPUT_SIZE)
Y = sinusoid_matrix(NUM_DATA, OUTPUT_SIZE)
# Y = torch.zeros(NUM_DATA, OUTPUT_SIZE)
# Y = checkered_matrix(NUM_DATA, OUTPUT_SIZE)
Y.requires_grad_(True)

trainset = torch.utils.data.TensorDataset(X, Y)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

# Model
print('==> Building model..')
kernel_size = 10
# random convolutional matrix with kernel length `kernel_size`
net = FullyConnectedNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

def convolutional_matrix(kernel, size):
    W = torch.zeros(size, size)
    for i in range(size):
        for k in range(0, len(kernel)):
            j = (i+k) % size
            W[i][j] = kernel[k]
    return W

def random_convolutional_matrix(kernel_size, size):
    kernel = torch.randn(kernel_size)
    return convolutional_matrix(kernel, size)
    
k1 = list([i - 5 for i in range(10)])
# random.shuffle(k1)
k2 = list([i - 5 for i in range(10)])
# random.shuffle(k2)
W1 = convolutional_matrix(k1, INPUT_SIZE)
W2 = convolutional_matrix(k2, INPUT_SIZE)

with torch.no_grad():
    net.linear_layers[0].weight.copy_(W1)
    net.linear_layers[0].bias.zero_()
    net.linear_layers[1].weight.copy_(W2)
    net.linear_layers[1].bias.zero_()

for param in net.parameters():
    print(param.shape)
    param.requires_grad = False

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
criterion = nn.MSELoss()
optimizer = optim.Adam([X, Y], lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def plot_weights(net, epoch):
    print(net.linear_layers)
    for i, layer in enumerate(net.linear_layers):
        W = layer.weight
        W = W.detach().cpu().numpy()
        plot_matrix(W, "Weight matrix at epoch %d" % epoch, os.path.join(save_dir, "weights_epoch_%d_layer_%d.png" % (epoch, i)))

def plot_data(X, name, epoch):
    X = X.detach().cpu().numpy()
    plt.imshow(X, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title("{} data at epoch {}".format(name, epoch))
    plt.savefig(os.path.join(save_dir, "{}_data_epoch_{}.png".format(name, epoch)), dpi=100)
    plt.clf()

def plot_matrix(M, title, filename):
    plt.imshow(M, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename, dpi=100)
    plt.clf()

def plot_loss(train_losses):
    plt.semilogy(train_losses)
    plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=100)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_loss = train_loss/(batch_idx+1)
    print('[%s] Training -- Loss: %.3f' % (timestamp, avg_loss))

    return avg_loss
    # plot the weight matrix using matplotlib

plot_data(X, "X", 0)
plot_data(Y, "Y", 0)
Y_hat = net(X)
plot_data(Y_hat, "Y_hat", 0)
plot_weights(net, 0)
train_losses = []
for epoch in range(num_epochs):
    loss = train(epoch)
    train_losses.append(loss)
    scheduler.step()
plot_weights(net, 199)
plot_data(X, "X", 199)
plot_data(Y, "Y", 199)
Y_hat = net(X)
plot_data(Y_hat, "Y_hat", 199)
plot_loss(train_losses)
