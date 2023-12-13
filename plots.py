import os
from matplotlib import pyplot as plt

def make_title(title, suffix):
    if suffix:
        title += " ({})".format(suffix)
    return title

def make_filename(name, suffix):
    filename = name
    if suffix:
        filename += "_{}".format(suffix)
    filename += ".svg"
    return filename

def plot_weights(net, name, suffix, save_dir):
    for i, layer in enumerate(net.linear_layers, 1):
        title = make_title("Weight matrix for layer {}".format(i), suffix)
        filename = make_filename("{}_layer_{}".format(name, i), suffix)
        plot_matrix(layer.weight, title, filename, save_dir)

def plot_data(X, name, suffix, save_dir):
    title = make_title("{} data".format(name), suffix)
    filename = make_filename(name, suffix)
    plot_matrix(X, title, filename, save_dir)

def plot_matrix(M, title, filename, save_dir):
    M = M.detach().cpu().numpy()
    plt.imshow(M, interpolation='nearest', aspect='auto')
    plt.colorbar()
    # Exclude titles from paper plots
    # plt.title(title)
    plt.savefig(os.path.join(save_dir, filename),bbox_inches='tight')
    plt.clf()

def plot_loss(train_losses, save_dir):
    plt.semilogy(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.savefig(os.path.join(save_dir, 'loss.svg'),bbox_inches='tight')
    plt.clf()
