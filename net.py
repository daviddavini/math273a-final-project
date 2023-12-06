import torch
import torch.nn as nn

from matrices import convolutional_matrix

class FullyConnectedNetwork(nn.Module):
    def __init__(self, *layer_sizes):
        super(FullyConnectedNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.layers = []
        self.linear_layers = []
        for i in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(linear)
            self.linear_layers.append(linear)
            self.layers.append(nn.ReLU())
        self.layers.pop()
        self.layers = nn.ModuleList(self.layers)

    def depth(self):
        return len(self.linear_layers)

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
def conv_net(width, kernels):
    layer_sizes = [width] * (len(kernels)+1)
    net = FullyConnectedNetwork(*layer_sizes)
    with torch.no_grad():
        for kernel, linear in zip(kernels, net.linear_layers):
            weight = convolutional_matrix(kernel, width)
            linear.weight.copy_(weight)
            linear.bias.zero_()
    return net

def random_conv_net(width, kernel_size, depth):
    return conv_net(width, [torch.randn(kernel_size) for _ in range(depth)])

def fully_connected_net(width, depth):
    layer_sizes = [width] * (depth+1)
    return FullyConnectedNetwork(*layer_sizes)

def conv_regularizer_constants(output_size, input_size, kernel_length):
    A = torch.zeros(output_size, input_size)
    for i in range(output_size):
        for j in range(input_size):
            A[i][j] = max(min(abs(i - j), input_size - abs(i - j)) - kernel_length, 0)
    return nn.Parameter(A, requires_grad=False)

class ConvolutionalRegularizer(nn.Module):
    def __init__(self, net, kernel_length, alpha):
        super(ConvolutionalRegularizer, self).__init__()
        self.net = net
        if self.net.depth() > 2:
            raise Exception("ConvolutionalRegularizer only works for depth 1 or 2 networks")
        self.alpha = alpha
        self.constants = nn.ParameterList()
        first_weight = self.net.linear_layers[0].weight
        first_constants = conv_regularizer_constants(first_weight.shape[0], first_weight.shape[1], kernel_length)
        self.constants.append(first_constants)
        if self.net.depth() == 2:
            second_weight = self.net.linear_layers[1].weight
            second_constants = conv_regularizer_constants(second_weight.shape[0], second_weight.shape[1], kernel_length)
            self.constants.append(second_constants)

    def forward(self):
        loss = torch.tensor(0.0, requires_grad=True).to(self.net.layers[0].weight.device)
        first_weight = self.net.linear_layers[0].weight
        loss = loss + torch.einsum("ij, ik, jk -> ", first_weight**2, first_weight**2, self.constants[0])
        if self.net.depth() == 2:
            second_weight = self.net.linear_layers[1].weight
            loss = loss + torch.einsum("ij, kj, ki -> ", second_weight**2, second_weight**2, self.constants[1])
        return self.alpha * loss