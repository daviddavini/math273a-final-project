import os
import torch
from torch.nn.functional import mse_loss
from matrices import sinusoid_matrix
from net import fully_connected_net, random_conv_net
from plots import plot_data, plot_loss, plot_weights
from sorting import sort_hidden_nodes_convolutionally
from utils import save_constants, setup_save_dir

SAVE_DIR = "images/learn_data/latest"
setup_save_dir(SAVE_DIR)

LEARNING_RATE = 1
NUM_EPOCHS = 10
NUM_DATA = 300
NET_WIDTH = 100
NET_DEPTH = 10
KERNEL_SIZE = 10

save_constants({
    "LEARNING_RATE": LEARNING_RATE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_DATA": NUM_DATA,
    "NET_WIDTH": NET_WIDTH,
    "NET_DEPTH": NET_DEPTH,
    "KERNEL_SIZE": KERNEL_SIZE
}, SAVE_DIR)

net = random_conv_net(NET_WIDTH, KERNEL_SIZE, NET_DEPTH)
for param in net.parameters():
    param.requires_grad = False

X = sinusoid_matrix(1, NUM_DATA, NET_WIDTH)
X.requires_grad = True
Y = torch.zeros(NUM_DATA, NET_WIDTH)
optimizer = torch.optim.Adam([X], lr=LEARNING_RATE)

plot_weights(net, "weights", None, SAVE_DIR)
plot_data(Y, "Y", None, SAVE_DIR)
# plot_data(X, "X", "start", SAVE_DIR)
losses = []
for epoch in range(NUM_EPOCHS):
    plot_data(X, "X", "epoch_{}".format(epoch), SAVE_DIR)
    optimizer.zero_grad()
    loss = mse_loss(net(X), Y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {:.5e}'.format(epoch, loss))

plot_loss(losses, SAVE_DIR)
# plot_data(X, "X", "end", SAVE_DIR)