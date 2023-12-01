import torch
import torch.nn as nn

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

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x

class ConvolutionalRegularizer(nn.Module):
    def __init__(self, net, kernel_length, alpha):
        super(ConvolutionalRegularizer, self).__init__()
        self.net = net
        self.alpha = alpha
        self.A_matrices = nn.ParameterList()
        for linear in net.linear_layers:
            A = torch.zeros(linear.weight.shape[1], linear.weight.shape[1])
            for i in range(A.shape[0]):
                for j in range(A.shape[0]):
                    A[i][j] = max(abs(i - j) - kernel_length, 0)
            self.A_matrices.append(nn.Parameter(A, requires_grad=False))

    def forward(self):
        loss = torch.tensor(0.0, requires_grad=True).to(self.net.layers[0].weight.device)
        for layer, A in zip(self.net.linear_layers, self.A_matrices):
            loss = loss + torch.einsum("ij, ik, jk -> ", layer.weight**2, layer.weight**2, A)
            # loss = loss + torch.einsum("ij, ik, jk -> ", torch.abs(layer.weight), torch.abs(layer.weight), A)
        return self.alpha * loss