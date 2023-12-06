import math
import torch


def checkered_matrix(m, n):
    M = torch.zeros(m, n)
    for i in range(m):
        for j in range(n):
            M[i][j] = (i % 2) ^ (j % 2)
    return M

def sinusoid_matrix(frequency, m, n):
    M = torch.zeros(m, n)
    for i in range(m):
        M[i][int((n//2 - 1) * math.sin(2 * math.pi * frequency * i / n)) + n//2] = 1.0
    return M

def line_matrix(slope, m, n):
    M = torch.zeros(m, n)
    for i in range(m):
        M[i][int(slope*i) % n] = 1.0
    return M

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