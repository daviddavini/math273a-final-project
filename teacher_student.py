import torch
from torch.nn.functional import mse_loss
from net import fully_connected_net, random_conv_net
from plots import plot_loss, plot_weights
from sorting import sort_hidden_nodes_convolutionally
from utils import save_constants, setup_save_dir

SAVE_DIR = "images/teacher_student/latest"
setup_save_dir(SAVE_DIR)

LEARNING_RATE = 1e-2
NUM_EPOCHS = 3000
NUM_DATA = 300
NET_WIDTH = 100
NET_DEPTH = 2
KERNEL_SIZE = 10

save_constants({
    "LEARNING_RATE": LEARNING_RATE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_DATA": NUM_DATA,
    "NET_WIDTH": NET_WIDTH,
    "NET_DEPTH": NET_DEPTH,
    "KERNEL_SIZE": KERNEL_SIZE
}, SAVE_DIR)

teacher = random_conv_net(NET_WIDTH, KERNEL_SIZE, NET_DEPTH)
for param in teacher.parameters():
    param.requires_grad = False
student = fully_connected_net(NET_WIDTH, NET_DEPTH)

optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)

plot_weights(student, "student_weights", "start", SAVE_DIR)
plot_weights(teacher, "teacher_weights", "start", SAVE_DIR)

losses = []
for epoch in range(NUM_EPOCHS):
    X = torch.randn(NUM_DATA, NET_WIDTH)
    Y = teacher(X)
    optimizer.zero_grad()
    loss = mse_loss(student(X), Y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {:.5e}'.format(epoch, loss))

plot_loss(losses, SAVE_DIR)
plot_weights(student, "student_weights", "end", SAVE_DIR)
plot_weights(teacher, "teacher_weights", "end", SAVE_DIR)

with torch.no_grad():
    sort_hidden_nodes_convolutionally(student, KERNEL_SIZE)
plot_weights(student, "student_weights_sorted", "end", SAVE_DIR)