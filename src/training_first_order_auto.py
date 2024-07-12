from first_order_optimizer import first_order_optimizer
from main_functions import *
import cProfile
import pstats
import matplotlib.pyplot as plt
from display_func import display
from network_architectures import digits_autoencoder_kfac, digits_classifier_kfac


# import random
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

#data_storage_device = "cpu"
data_storage_device = "cuda:0"
# train_device = "cpu"
train_device = "cuda:0"

problem_type = 0
run_test = True
verbose = 2 #1 means every batch, 2 means every epoch, 3 means only when done training, 4 means never.

train_data_location = "data/digits_train.npy"
train_target_location = "data/digits_train_targets.npy"
test_data_location = "data/digits_test.npy"
test_target_location = "data/digits_test_targets.npy"

# train_data_location = "data/cifar_rgb_train.npy"
# train_target_location = "data/cifar_rgb_train_targets.npy"
# test_data_location = "data/cifar_rgb_test.npy"
# test_target_location = "data/cifar_rgb_test_targets.npy"

# hyper params
num_epochs = 100
batch_size = 200

lr = 0.5
momen_max = 0.9  # if set to 0 there will be no momentum
momen_rate = None  # only needed for kfac but is needed as an argument in order to use the same functions
momen_conv = "constant" # "linear, "exp", "constant", "paper". It determines how the momentum will converve uopon start up
time_limit = 1E6
update_limit = 1E6

d = load_data(train_data_location, train_target_location, batch_size, data_storage_device,
              run_test, problem_type, test_data_location, test_target_location)

net = digits_classifier_kfac if problem_type else digits_autoencoder_kfac
net, loss_func, num_layers = construct_network(net,
                                               problem_type,
                                               train_device,
                                               #net_struct = [3000, 2000, 1500, 1000 ,250, 10],
                                               input_size=d["num_attributes"],
                                               bias=True)

optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momen_max, nesterov = True)


losses, t, net = first_order_optimizer(net, loss_func, optimizer, problem_type, num_epochs, lr, d, batch_size,
                                       momen_max, momen_conv, momen_rate, train_device, run_test, verbose,
                                       time_limit, update_limit)


if run_test:
    if not problem_type: display(d["test"], net.forward(d["test"]), num_examples=10, close=False)
    plt.plot(losses["test"])
    plt.show()

else:
    if not problem_type: display(d["train"], net.forward(d["train"]), num_examples=10, close=False)
    plt.plot(losses["train"])
    plt.show()




