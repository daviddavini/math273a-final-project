import torch.nn as nn

class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x