from kfac_optimizer_regu import kfac_optimizer
from main_functions import *
import cProfile
import pstats
import matplotlib.pyplot as plt
from display_func import display
from network_architectures import digits_autoencoder_kfac, digits_classifier_kfac, drop_out_digits_classifier_kfac


# import random
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

data_storage_device = "cpu"
#data_storage_device = "cuda:0"
# train_device = "cpu"
train_device = "cuda:0"

problem_type = 1
run_test = True
verbose = 1

# train_data_location = "data/digits_train.npy"
# train_target_location = "data/digits_train_targets.npy"
# test_data_location = "data/digits_test.npy"
# test_target_location = "data/digits_test_targets.npy"

train_data_location = "data/cifar_rgb_train.npy"
train_target_location = "data/cifar_rgb_train_targets.npy"
test_data_location = "data/cifar_rgb_test.npy"
test_target_location = "data/cifar_rgb_test_targets.npy"



# hyper params
num_epochs = 100
batch_size = 10000
desired_batch_size = 10000

lr = 0.5
scaling_1 = 3 # 3 is prolly the best option since only 3 works for classification
scaling_2 = 3 # 3 is the only viable option
momen_max = 0.75  # if set to 0 there will be no momentum
momen_rate = 0.25  # between 0 and 1, the greater the value the faster the max momentum will be reached.
momen_conv = "linear"  # "linear, "exp", "constant", "paper". It determines how the momentum will converve uopon start up
time_limit = 1E6
update_limit = 1E6
l2_scaling = 0.001

d = load_data(train_data_location, train_target_location, batch_size, data_storage_device,
              run_test, problem_type, test_data_location, test_target_location)

net = digits_classifier_kfac if problem_type else digits_autoencoder_kfac
net, loss_func, num_layers = construct_network(net, problem_type, train_device,
                                               net_struct = [3000, 2000, 1500, 1000 ,250, 10],
                                               input_size=int(d["num_attributes"]))


losses, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                desired_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                train_device, run_test, verbose, time_limit, update_limit, l2_scaling)


# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats(10)
translabels = {0:"plane",1:"car", 2:"bird", 3:"cat", 4:"deer",
               5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

if problem_type:
    display(d["test"],  num_examples=20, close=False, image_dim=32)
    print(torch.argmax(net.forward(d["test"][:20]), dim=1))
    print(d["test_targets"][:20])

else:
    display(d["test"], net.forward(d["test"]), num_examples=10, close=False, image_dim=32)

plt.plot(losses["test"])
plt.show()




