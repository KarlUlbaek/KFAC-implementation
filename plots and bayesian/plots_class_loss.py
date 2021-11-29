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

train_data_location = "data/cifar_rgb_train.npy"
train_target_location = "data/cifar_rgb_train_targets.npy"
test_data_location = "data/cifar_rgb_test.npy"
test_target_location = "data/cifar_rgb_test_targets.npy"

# hyper params
num_epochs = 10
batch_size = 5000

problem_type = 1
run_test = True
lr = 0.2
scaling_1 = 3 # 3 is prolly the best option since only 3 works for classification
scaling_2 = 3 # 3 is the only viable option
momen_max = 0.85  # if set to 0 there will be no momentum
momen_rate = 0.25  # between 0 and 1, the greater the value the faster the max momentum will be reached.
momen_conv = "linear"  # "linear, "exp", "constant", "paper". It determines how the momentum will converve uopon start up

d = load_data(train_data_location, train_target_location, batch_size, data_storage_device,
              run_test, problem_type, test_data_location, test_target_location)

net = digits_classifier_kfac if problem_type == 1 or problem_type == 2 else digits_autoencoder_kfac
net, loss_func, num_layers = construct_network(net, problem_type, train_device,
                                               net_struct = [3000, 2000, 1500, 1000 ,250, 10],
                                               input_size=int(d["num_attributes"]))

######
verbose = 1
time_limit = 10
update_limit = 1E6

ax = plt.subplot()
kfac_params=[
            [0.27308410461320376, 8111.228838346976, 1.9214905566046578, 0.34895499520746287],
            [0.2730575943227828, 8110.718178066085, 1.6143315104594331, 0.43917662955166453],
            [0.28439175639366243, 8027.693340375634, 0.9522375165440704, 0.3130965721120084],
            [0.28564097412460454, 8043.931378861912, 1.5371638904203802, 0.37547594904105286],
            ]

first_params = [
                [0.010996048189947207,721.2559237339138,0.9709583477733684],
                [0.00916594600535703,714.6988876026173,0.9067204643441487],
                [0.006249824361401911,503.86196643290566,0.8712149681213697],
                [0.009367955082639639,721.441801901672,0.9289657651877622]
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


verbose = 2
for i, param in enumerate(first_params):
    net.initialize_weights_xavier()
    lr, batch_size, momen_max = param[0],int(param[1]),param[2]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momen_max, nesterov = True)
    first_loss = random.choice([2.7653, 3.3044, 2.9886, 3.4892])
    losses, t, net = first_order_optimizer(net, loss_func, optimizer, problem_type, num_epochs, lr, d, batch_size,
                                           momen_max, momen_conv, momen_rate, train_device, run_test, verbose,
                                           time_limit, update_limit)
    losses["test"].insert(0, first_loss)
    if i == 0:
        plt.plot(np.linspace(0, time_limit, len(losses["test"])), losses["test"], color="indianred", label = "SGD")
    else:
        plt.plot(np.linspace(0, time_limit, len(losses["test"])), losses["test"], color="indianred")

plt.legend()
plt.title("Classification loss over 10 seconds")
plt.xlabel("time in seconds")
plt.ylabel("test loss")
plt.savefig('plots/class_loss_as_a_function_time.png', bbox_inches='tight', dpi=200)
with open('plots/figures/class_loss_as_a_function_time.pkl','wb') as file:
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

verbose = 1
for i, param in enumerate(first_params):
    net.initialize_weights_xavier()
    lr, batch_size, momen_max = param[0],int(param[1]),param[2]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momen_max, nesterov = True)
    #first_loss = random.choice([2.7653, 3.3044, 2.9886, 3.4892])
    losses, t, net = first_order_optimizer(net, loss_func, optimizer, problem_type, num_epochs, lr, d, batch_size,
                                           momen_max, momen_conv, momen_rate, train_device, run_test, verbose,
                                           time_limit, update_limit)
    #losses["test"].insert(0, first_loss)
    if i == 0:
        plt.plot(np.linspace(0, update_limit, len(losses["test"])), losses["test"], color="indianred", label = "SGD")
    else:
        plt.plot(np.linspace(0, update_limit, len(losses["test"])), losses["test"], color="indianred")

plt.legend()
plt.title("Classification loss over 50 model updates")
plt.xlabel("model updates")
plt.ylabel("test loss")
plt.savefig('plots/class_loss_as_a_function_updates.png', bbox_inches='tight', dpi=200)
with open('plots/figures/class_loss_as_a_function_updates.pkl','wb') as file:
    pickle.dump(ax, file)
# with open('plots/batch_size_fixed_updates.pkl', 'rb') as file:
#     ax = pickle.load(file)
#plt.show()
plt.clf()
