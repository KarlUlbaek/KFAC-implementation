from kfac_optimizer import kfac_optimizer
from first_order_optimizer import first_order_optimizer
from main_functions import *
import cProfile
import pstats
import matplotlib.pyplot as plt
from display_func import display
from network_architectures import digits_autoencoder_kfac, digits_classifier_kfac
plt.style.use('ggplot')
import pickle
import random

# import random
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

data_storage_device = "cpu"
# data_storage_device = "cuda:0"
# train_device = "cpu"
train_device = "cuda:0"

train_data_location = "data/slice_train.npy"
train_target_location = "data/slice_train_targets.npy"
test_data_location = "data/slice_test.npy"
test_target_location = "data/slice_test_targets.npy"

# hyper params
num_epochs = 10
batch_size = 5000

problem_type = 2
run_test = True
lr = 0.2
scaling_1 = 3 # 3 is prolly the best option since only 3 works for classification
scaling_2 = 3 # 3 is the only viable option
momen_max = 0.85  # if set to 0 there will be no momentum
momen_rate = 0.25  # between 0 and 1, the greater the value the faster the max momentum will be reached.
momen_conv = "linear"  # "linear, "exp", "constant", "paper". It determines how the momentum will converve uopon start up

d = load_data(train_data_location, train_target_location, batch_size, data_storage_device,
              run_test, problem_type, test_data_location, test_target_location, scaling=1)

d["test_targets"] = d["test_targets"] / torch.max(d["train_targets"])
d["train_targets"] = d["train_targets"] / torch.max(d["train_targets"])


net = digits_classifier_kfac if problem_type == 1 or problem_type == 2 else digits_autoencoder_kfac
net, loss_func, num_layers = construct_network(net, problem_type, train_device,
                                               net_struct=[500, 750, 500, 250, 100, 1],
                                               input_size=int(d["num_attributes"]))

######
verbose = 1
time_limit = 5
update_limit = 1E6

ax = plt.subplot()
kfac_params=[
    [0.03917475852020402, 13869.147068028093, 1.1172765848993051, 0.730883758100982],
    [0.04142371375700023, 12236.745066588674, 0.5753376242935113, 0.8381467056427245],
    [0.13943893101012308, 13433.106651155766, 0.571803524638441, 0.9379686266214364],
    [0.15151488221762666, 12727.956247513072, 0.9036091941915407, 0.7515518769482504],
       ]

for i, param in enumerate(kfac_params):
    net.initialize_weights_xavier()
    lr, batch_size, momen_rate, momen_max = param[0],int(param[1]),param[2],param[3]
    desired_batch_size = batch_size
    losses, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                    desired_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                    train_device, run_test, verbose, time_limit, update_limit)
    if i == 0:
        plt.plot(np.linspace(0, time_limit, len(losses["test"])), losses["test"], color="cornflowerblue", label = "KFAC")
    else:
        plt.plot(np.linspace(0, time_limit, len(losses["test"])), losses["test"], color="cornflowerblue")

first_params = [
    [0.00638289578512369, 3523.5110264992804, 0.99],
    [0.0060270113920994, 588.4240246847546, 0.9019332823662084],
    [0.010437899065995144, 595.862887288769, 0.8382426045432513],
    [0.013776641491279015, 1789.5543376406831, 0.8120175298798354],
    ]

verbose = 2
for i, param in enumerate(first_params):
    net.initialize_weights_xavier()
    lr, batch_size, momen_max = param[0],int(param[1]),param[2]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momen_max, nesterov = True)
    first_loss = loss_func( torch.squeeze(net.forward(d["test"].cuda())), d["test_targets"].cuda()).detach().cpu()
    #first_loss = random.choice([0.09, 0.1, 0.11])
    losses, t, net = first_order_optimizer(net, loss_func, optimizer, problem_type, num_epochs, lr, d, batch_size,
                                           momen_max, momen_conv, momen_rate, train_device, run_test, verbose,
                                           time_limit, update_limit)
    losses["test"].insert(0, first_loss)
    if i == 0:
        plt.plot(np.linspace(0, time_limit, len(losses["test"])), losses["test"], color="indianred", label = "SGD")
    else:
        plt.plot(np.linspace(0, time_limit, len(losses["test"])), losses["test"], color="indianred")

plt.legend()
plt.title("Regression loss over 5 seconds")
plt.xlabel("time in seconds")
plt.ylabel("test loss")
plt.ylim(-0.005, 0.05)
plt.savefig('plots/slice_loss_as_a_function_time.png', bbox_inches='tight', dpi=200)
with open('plots/figures/slice_loss_as_a_function_time.pkl', 'wb') as file:
    pickle.dump(ax, file)
# with open('plots/batch_size_fixed_updates.pkl', 'rb') as file:
#     ax = pickle.load(file)
#plt.show()
plt.clf()



######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
ax = plt.subplot()
verbose = 1
time_limit = 10000
update_limit = 50

kfac_params=[
    [0.03917475852020402, 13869.147068028093, 1.1172765848993051, 0.730883758100982],
    [0.04142371375700023, 12236.745066588674, 0.5753376242935113, 0.8381467056427245],
    [0.13943893101012308, 13433.106651155766, 0.571803524638441, 0.9379686266214364],
    [0.15151488221762666, 12727.956247513072, 0.9036091941915407, 0.7515518769482504],
       ]

for i, param in enumerate(kfac_params):
    net.initialize_weights_xavier()
    lr, batch_size, momen_rate, momen_max = param[0],int(param[1]),param[2],param[3]
    desired_batch_size = batch_size
    losses, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                    desired_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                    train_device, run_test, verbose, time_limit, update_limit)

    if i == 0:
        plt.plot(np.linspace(0, update_limit, len(losses["test"])), losses["test"], color="cornflowerblue", label = "KFAC")
    else:
        plt.plot(np.linspace(0, update_limit, len(losses["test"])), losses["test"], color="cornflowerblue")

first_params = [
    [0.00638289578512369, 3523.5110264992804, 0.99],
    [0.0060270113920994, 588.4240246847546, 0.9019332823662084],
    [0.010437899065995144, 595.862887288769, 0.8382426045432513],
    [0.013776641491279015, 1789.5543376406831, 0.8120175298798354],
    ]


verbose = 1
for i, param in enumerate(first_params):
    net.initialize_weights_xavier()
    lr, batch_size, momen_max = param[0],int(param[1]),param[2]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momen_max, nesterov = True)
    first_loss = loss_func( torch.squeeze(net.forward(d["test"].cuda())), d["test_targets"].cuda()).detach().cpu()
    #first_loss = random.choice([0.09, 0.1, 0.11])
    losses, t, net = first_order_optimizer(net, loss_func, optimizer, problem_type, num_epochs, lr, d, batch_size,
                                           momen_max, momen_conv, momen_rate, train_device, run_test, verbose,
                                           time_limit, update_limit)
    losses["test"].insert(0, first_loss)
    if i == 0:
        plt.plot(np.linspace(0, update_limit, len(losses["test"])), losses["test"], color="indianred", label = "SGD")
    else:
        plt.plot(np.linspace(0, update_limit, len(losses["test"])), losses["test"], color="indianred")

plt.legend()
plt.title("Regression loss over 50 model updates")
plt.xlabel("model updates")
plt.ylabel("test loss")
plt.ylim(-0.005, 0.05)
plt.savefig('plots/slice_loss_as_a_function_updates.png', bbox_inches='tight', dpi=200)
with open('plots/figures/slice_loss_as_a_function_updates.pkl', 'wb') as file:
    pickle.dump(ax, file)
# with open('plots/batch_size_fixed_updates.pkl', 'rb') as file:
#     ax = pickle.load(file)
#plt.show()
plt.clf()
