from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse

from net import ConvolutionalRegularizer, FullyConnectedNetwork
import datetime

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
best_acc = 0  # best test accuracy

# Data
print('==> Preparing data..')
# Generate random input/output vector pairs
INPUT_SIZE = 32
OUTPUT_SIZE = 32
TRAIN_SIZE = 1000
TEST_SIZE = 100
x_train = torch.randn(TRAIN_SIZE, INPUT_SIZE)
# Q = torch.randn(INPUT_SIZE, INPUT_SIZE, OUTPUT_SIZE)
# b = torch.randn(OUTPUT_SIZE, INPUT_SIZE)
# y = torch.einsum("ij, ik, jkl -> il", x, x, Q) + torch.einsum("ij, kj -> ik", x, b)
A = torch.randn(OUTPUT_SIZE, INPUT_SIZE)
y_train = x_train @ A.T
trainset = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
x_test = torch.randn(TEST_SIZE, INPUT_SIZE)
y_test = x_test @ A.T
testset = torch.utils.data.TensorDataset(x_test, y_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# Model
print('==> Building model..')
net = FullyConnectedNetwork(INPUT_SIZE, OUTPUT_SIZE)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
criterion = nn.MSELoss()
# regularizer = ConvolutionalRegularizer(net)
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    # momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def plot_weights(net, epoch):
    W = net.layers[0].weight
    W = W.detach().cpu().numpy()
    plt.imshow(W, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('Weight matrix at epoch %d' % epoch)
    plt.savefig('weight_epoch_%d.png' % epoch, dpi=100)
    plt.clf()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets) #+ regularizer()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_loss = train_loss/(batch_idx+1)
    print('[%s] Training -- Loss: %.3f' % (timestamp, avg_loss))

    return avg_loss
    # plot the weight matrix using matplotlib

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('[%s] Testing  -- Loss: %.3f' % (timestamp, test_loss/(batch_idx+1)))

    return test_loss/(batch_idx+1)

plot_weights(net, 0)
train_losses = []
test_losses = []
for epoch in range(200):
    loss = train(epoch)
    train_losses.append(loss)
    loss = test(epoch)
    test_losses.append(loss)
    scheduler.step()
plot_weights(net, 199)
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.savefig('loss.png', dpi=100)