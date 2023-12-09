# Checking that our solution to the linear regression problem is correct
# (Or, at least, that it is a local minimums)

import torch

NUM_DATA = 300
NET_WIDTH = 100

X = torch.randn(NUM_DATA, NET_WIDTH)
Y = torch.randn(NUM_DATA, NET_WIDTH)
W = Y.T @ X @ torch.inverse(X.T @ X)
loss = (Y - X @ W).norm()
print(loss)

min_loss = 1e100
for i in range(1000):
    W2 = W + torch.randn(W.shape) * 0.01
    loss2 = (Y - X @ W2).norm()
    if loss2 < min_loss:
        min_loss = loss2
print("%.9f" % (min_loss/loss).item(), end=" ")
