# Goal: Learn X that creates a convolutional W in linear regression setting
import torch
from matrices import checkered_matrix, line_matrix, random_convolutional_matrix, sinusoid_matrix
from plots import plot_data, plot_loss
from utils import save_constants, setup_save_dir

SAVE_DIR = "images/projection/latest"
setup_save_dir(SAVE_DIR)

LEARNING_RATE = 1e-3
NUM_EPOCHS = 10000
NUM_DATA = 100
NET_WIDTH = 100
KERNEL_SIZE = 10
TARGET_MEAN_SQUARE = 10
ZERO_WEIGHT = 1
OFF_DIAGONAL_WEIGHT = 0

save_constants({
    "LEARNING_RATE": LEARNING_RATE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_DATA": NUM_DATA,
    "NET_WIDTH": NET_WIDTH,
    "KERNEL_SIZE": KERNEL_SIZE
}, SAVE_DIR)

# X = checkered_matrix(NUM_DATA, NET_WIDTH)
X = line_matrix(1, NUM_DATA, NET_WIDTH)
X.requires_grad = True
# Y = torch.randn(NUM_DATA, NET_WIDTH)
Y_preimage = sinusoid_matrix(1, NUM_DATA, NET_WIDTH)
W = random_convolutional_matrix(KERNEL_SIZE, NET_WIDTH)
Y = Y_preimage @ W

optimizer = torch.optim.Adam([X], lr=LEARNING_RATE)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS // 2)

A = torch.zeros(NET_WIDTH, NET_WIDTH)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if j >= i and j < i + KERNEL_SIZE:
            A[i][j] = 1.0

def solve_linear_regression(X, Y):
    '''
    Returns W, the minimizer of ||Y - XW||^2
    Where X has shape (num_data, net_width)
    and Y has shape (num_data, net_width)
    '''
    return Y.T @ X @ torch.pinverse(X.T @ X)
    # return Y.T @ X @ torch.inverse(X.T @ X)
    # W, _ = torch.solve(X.T @ Y, X.T @ X)
    # return W

def conv_loss(W, zero_weight, off_diagonal_weight):
    zeros_penalty = ((W * (1 - A)) ** 2).mean() / 2
    off_diagonal_penalty = torch.tensor(0)
    for i in range(KERNEL_SIZE):
        off_diagonal_penalty = off_diagonal_penalty + torch.var(torch.diag(W, diagonal=i))
    return zeros_penalty * zero_weight + off_diagonal_penalty * off_diagonal_weight

plot_data(Y, "Y", None, SAVE_DIR)
plot_data(X, "X", "start", SAVE_DIR)
plot_data(W, "W", None, SAVE_DIR)
W_star = solve_linear_regression(X, Y)
plot_data(W_star, "W_star", "start", SAVE_DIR)
losses = []
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    W_star = solve_linear_regression(X, Y)
    # loss = ((Y - X @ W_star) ** 2).mean() / 2
    loss = conv_loss(W_star, ZERO_WEIGHT, OFF_DIAGONAL_WEIGHT)
    loss = loss + torch.relu(TARGET_MEAN_SQUARE - (W_star ** 2).mean())
    losses.append(loss.item())
    loss.backward()
    torch.nn.utils.clip_grad_norm_([X], 1e-1)
    optimizer.step()
    # scheduler.step()
    print('Epoch: {}, Loss: {:.5e}'.format(epoch, loss))

plot_loss(losses, SAVE_DIR)
plot_data(X, "X", "end", SAVE_DIR)
W_star = solve_linear_regression(X, Y)
plot_data(W_star, "W_star", "end", SAVE_DIR)