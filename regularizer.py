import torch
from torch.nn.functional import mse_loss
from matrices import line_matrix
from net import fully_connected_net, random_conv_net, ConvolutionalRegularizer
from plots import plot_data, plot_loss, plot_weights
from sorting import sort_hidden_nodes_convolutionally
from utils import save_constants, setup_save_dir

SAVE_DIR = "images/regularizer/latest"
setup_save_dir(SAVE_DIR)

LEARNING_RATE = 1e-3
NUM_EPOCHS = 3000
NUM_DATA = 300
NET_WIDTH = 100
NET_DEPTH = 2
KERNEL_SIZE = 10
REG_WEIGHT = 1

save_constants({
    "LEARNING_RATE": LEARNING_RATE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_DATA": NUM_DATA,
    "NET_WIDTH": NET_WIDTH,
    "NET_DEPTH": NET_DEPTH,
    "KERNEL_SIZE": KERNEL_SIZE
}, SAVE_DIR)

net = fully_connected_net(NET_WIDTH, NET_DEPTH)

X = torch.randn(NUM_DATA, NET_WIDTH)
A = torch.randn(NET_WIDTH, NET_WIDTH)
Y = X @ A.T
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

regularizer = ConvolutionalRegularizer(net, KERNEL_SIZE, alpha=REG_WEIGHT)

plot_weights(net, "weights", "start", SAVE_DIR)
plot_data(Y, "Y", None, SAVE_DIR)
plot_data(X, "X", None, SAVE_DIR)
losses = []
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    loss = mse_loss(net(X), Y) + regularizer()
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {:.5e}'.format(epoch, loss))

plot_loss(losses, SAVE_DIR)
plot_weights(net, "weights", "end", SAVE_DIR)

with torch.no_grad():
    sort_hidden_nodes_convolutionally(net, KERNEL_SIZE)
plot_weights(net, "weights_sorted", "end", SAVE_DIR)