import os
from matplotlib import pyplot as plt


def plot_weights(net, name, suffix, save_dir):
    for i, layer in enumerate(net.linear_layers, 1):
        title = "Weight matrix for layer {} ({})".format(i, suffix)
        filename = "{}_layer_{}_{}.png".format(name, i, suffix)
        plot_matrix(layer.weight, title, filename, save_dir)

def plot_data(X, name, suffix, save_dir):
    title = "{} data ({})".format(name, suffix)
    filename = "{}_{}.png".format(name, suffix)
    plot_matrix(X, title, filename, save_dir)

def plot_matrix(M, title, filename, save_dir):
    M = M.detach().cpu().numpy()
    plt.imshow(M, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.clf()

def plot_loss(train_losses, save_dir):
    plt.semilogy(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=300)
    plt.clf()
