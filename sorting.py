import torch

def argsort_rows_by_moment(weight):
    row_indices = torch.arange(weight.shape[1])
    weighted_indices = torch.einsum("ij, j -> ij", torch.abs(weight), row_indices) 
    row_sums = weight.abs().sum(axis=1)
    avg_indices = torch.einsum("ij, i -> i", weighted_indices, 1/row_sums)
    row_order = torch.argsort(avg_indices)
    return row_order

def sort_hidden_nodes_by_moment(student):
    for layer_1, layer_2 in zip(student.linear_layers, student.linear_layers[1:]):
        row_order = argsort_rows_by_moment(layer_1.weight)
        layer_1.weight = torch.nn.Parameter(layer_1.weight[row_order])
        layer_2.weight = torch.nn.Parameter(layer_2.weight[:, row_order])

def argsort_rows_convolutionally(weight, kernel_size, eps=0.1):
    weight = weight.clone()
    weight[:, :kernel_size] = 0
    first_nonzero = torch.argmax((weight.abs() > eps).float(), axis=1)
    return first_nonzero.argsort()

def sort_hidden_nodes_convolutionally(student, kernel_size):
    for layer_1, layer_2 in zip(student.linear_layers, student.linear_layers[1:]):
        row_order = argsort_rows_convolutionally(layer_1.weight, kernel_size)
        layer_1.weight = torch.nn.Parameter(layer_1.weight[row_order])
        layer_2.weight = torch.nn.Parameter(layer_2.weight[:, row_order])
