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

train_data_location = "data/digits_train.npy"
train_target_location = "data/digits_train_targets.npy"
test_data_location = "data/digits_test.npy"
test_target_location = "data/digits_test_targets.npy"

# hyper params
num_epochs = 10
batch_size = 5000

lr = 0.2
scaling_1 = 3 # 3 is prolly the best option since only 3 works for classification
scaling_2 = 3 # 3 is the only viable option
momen_max = 0.85  # if set to 0 there will be no momentum
momen_rate = 0.25  # between 0 and 1, the greater the value the faster the max momentum will be reached.
momen_conv = "linear"  # "linear, "exp", "constant", "paper". It determines how the momentum will converve uopon start up

d = load_data(train_data_location, train_target_location, batch_size, data_storage_device,
              run_test, problem_type, test_data_location, test_target_location)

net = digits_classifier_kfac if problem_type else digits_autoencoder_kfac
net, loss_func, num_layers = construct_network(net, problem_type, train_device)

kfac_params=[
        [0.18091478519897453,9543.235910696467,0.5279184525860896,0.6066011650262455],
        [0.19801299042413312,10109.387282268859,0.7789782020549842,0.5446865337129331],
        [0.14375623156253733,4419.346043731717,1.1906234050193716,0.5560639333641116],
        [0.1594774014812677,4765.805432251613,1.762765115143839,0.3935509738186864]
       ]

#
# ######
ax = plt.subplot()
verbose = 3
time_limit = 1000
update_limit = 50

batch_sizes = [100, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000]
for i, param in enumerate(kfac_params):
    losses = []
    lr, batch_size, momen_rate, momen_max = param[0],int(param[1]),param[2],param[3]
    for batch_size in batch_sizes:
        net.initialize_weights_xavier()
        print("model ", i+1, "batch size: ", batch_size)
        desired_batch_size = batch_size
        loss, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                        desired_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                        train_device, run_test, verbose, time_limit, update_limit)
        losses.append(loss["test"][-1])


    plt.plot(batch_sizes, losses, color="cornflowerblue")


plt.legend()
plt.title("Impact of batch size for fixed number of updates")
plt.ylim(0, 0.03)
plt.xlabel("batch size")
plt.ylabel("test loss")
plt.savefig('plots/batch_size_fixed_updates.png', bbox_inches='tight', dpi=200)
with open('plots/figures/batch_size_fixed_updates.pkl','wb') as file:
    pickle.dump(ax, file)
plt.clf()
# with open('plots/batch_size_fixed_updates.pkl', 'rb') as file:
#     ax = pickle.load(file)
#plt.show()
#
#
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
#
#
#
#
ax = plt.subplot()
verbose = 3
time_limit = 10
update_limit = 5000

batch_sizes = [100, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000]
for i, param in enumerate(kfac_params):
    losses = []
    lr, batch_size, momen_rate, momen_max = param[0],int(param[1]),param[2],param[3]
    for batch_size in batch_sizes:
        net.initialize_weights_xavier()
        print("model ", i+1, "batch size: ", batch_size)
        desired_batch_size = batch_size
        loss, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                        desired_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                        train_device, run_test, verbose, time_limit, update_limit)
        losses.append(loss["test"][-1])


    plt.plot(batch_sizes, losses, color="cornflowerblue")


plt.legend()
plt.title("Impact of batch size for fixed training time")
plt.xlabel("batch size")
plt.ylabel("test loss")
plt.ylim(0, 0.03)
plt.savefig('plots/batch_size_fixed_time.png', bbox_inches='tight', dpi=200)
with open('plots/figures/batch_size_fixed_time.pkl','wb') as file:
    pickle.dump(ax, file)
plt.clf()
# with open('plots/batch_size_fixed_updates.pkl', 'rb') as file:
#     ax = pickle.load(file)
#plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

ax = plt.subplot()
verbose = 3
time_limit = 10
update_limit = 5000

kfac_params = kfac_params[0]
batch_sizes = [25, 50, 100, 250, 500, 750, 1000]
lr, batch_size, momen_rate, momen_max = kfac_params[0],int(kfac_params[1]),kfac_params[2],kfac_params[3]
lrs = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
colors = ["cornflowerblue", "indianred", "orangered", "seagreen", "palevioletred", "mediumpurple"]
for i, lr in enumerate(lrs):
    losses = []
    for batch_size in batch_sizes:
        net.initialize_weights_xavier()
        print("model ", i+1, "batch size: ", batch_size)
        desired_batch_size = batch_size
        loss, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                        desired_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                        train_device, run_test, verbose, time_limit, update_limit)
        losses.append((loss["test"][-1]))


    plt.plot(batch_sizes, losses, color=colors[i], label = "LR: "+str(lr))


plt.legend()
plt.title("Learning from small batches adjusted for LR")
plt.xlabel("batch size")
plt.ylabel("test loss")
plt.ylim(0, 0.15)
plt.savefig('plots/small_batch_size_fixed_time.png', bbox_inches='tight', dpi=200)
with open('plots/figures/small_batch_size_fixed_time.pkl','wb') as file:
    pickle.dump(ax, file)
plt.clf()
with open('plots/batch_size_fixed_updates.pkl', 'rb') as file:
    ax = pickle.load(file)
plt.show()
#
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
#
#
#
#
#
#
######
ax = plt.subplot()
verbose = 3
time_limit = 1000
update_limit = 3

kfac_params=[[0.18091478519897453,9543.235910696467,0.5279184525860896,0.6066011650262455]]
batch_sizes = [50, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000]
for i, param in enumerate(kfac_params):
    losses = []
    lr, batch_size, momen_rate, momen_max = param[0],int(param[1]),param[2],param[3]
    for batch_size in batch_sizes:
        net.initialize_weights_xavier()
        print("model ", i+1, "batch size: ", batch_size)
        desired_batch_size = batch_size
        loss, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                        desired_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                        train_device, run_test, verbose, time_limit, update_limit)
        losses.append(t["inv_time"]/t["update_time"][1])


    plt.plot(batch_sizes, losses, color="cornflowerblue")


plt.legend()
plt.title("Fraction of time spend inverting")
#plt.ylim(0, 0.05)
plt.xlabel("batch size")
plt.ylabel("fraction")
plt.savefig('plots/inversion_fraction.png', bbox_inches='tight', dpi=200)
with open('plots/figures/inversion_fraction.pkl','wb') as file:
    pickle.dump(ax, file)
plt.clf()
# plt.show()
#
#
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
#
kfac_params=[
        [0.18091478519897453,9543.235910696467,0.5279184525860896,0.6066011650262455],
        [0.19801299042413312,10109.387282268859,0.7789782020549842,0.5446865337129331],
        [0.14375623156253733,4419.346043731717,1.1906234050193716,0.5560639333641116],
        [0.1594774014812677,4765.805432251613,1.762765115143839,0.3935509738186864]
       ]


ax = plt.subplot()
verbose = 3
time_limit = 10
update_limit = 5000

artificial_batch_sizes = [100, 200, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000]
batch_size = 25

colors = ["cornflowerblue", "indianred", "orangered", "seagreen", "palevioletred", "mediumpurple"]
for i, param in enumerate(kfac_params):
    losses = []
    lr, _, momen_rate, momen_max = param[0],int(param[1]),param[2],param[3]
    for artificial_batch_size in artificial_batch_sizes:
        net.initialize_weights_xavier()
        print("model ", i+1, "artificial_batch_size size: ", artificial_batch_size)
        loss, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                      artificial_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                      train_device, run_test, verbose, time_limit, update_limit)
        losses.append(loss["test"][-1])


    plt.plot(artificial_batch_sizes, losses, color="cornflowerblue")

plt.legend()
plt.title("Impact of artificially increasing batch size \n for fixed training time")
plt.xlabel("artificial batch size")
plt.ylabel("test loss")
plt.ylim(0, 0.1)
plt.savefig('plots/artificial_batch_size.png', bbox_inches='tight', dpi=200)
with open('plots/figures/artificial_batch_size.pkl', 'wb') as file:
    pickle.dump(ax, file)
plt.show()
plt.clf()
with open('plots/figures/artificial_batch_size.pkl', 'rb') as file:
    ax = pickle.load(file)
plt.show()


# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
#
kfac_params=[
        [0.18091478519897453,9543.235910696467,0.5279184525860896,0.6066011650262455],
        [0.19801299042413312,10109.387282268859,0.7789782020549842,0.5446865337129331],
        [0.14375623156253733,4419.346043731717,1.1906234050193716,0.5560639333641116],
        [0.1594774014812677,4765.805432251613,1.762765115143839,0.3935509738186864]
       ]


ax = plt.subplot()
verbose = 3
time_limit = 1000
update_limit = 50

artificial_batch_sizes = [100, 200, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000]
batch_size = 25

colors = ["cornflowerblue", "indianred", "orangered", "seagreen", "palevioletred", "mediumpurple"]
for i, param in enumerate(kfac_params):
    losses = []
    lr, _, momen_rate, momen_max = param[0],int(param[1]),param[2],param[3]
    for artificial_batch_size in artificial_batch_sizes:
        net.initialize_weights_xavier()
        print("model ", i+1, "artificial_batch_size size: ", artificial_batch_size)
        loss, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                      artificial_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                      train_device, run_test, verbose, time_limit, update_limit)
        losses.append(loss["test"][-1])


    plt.plot(artificial_batch_sizes, losses, color="cornflowerblue")

plt.legend()
plt.title("Impact of artificially increasing batch size \n for fixed number of updates")
plt.xlabel("artificial batch size")
plt.ylabel("test loss")
plt.ylim(0, 0.1)
plt.savefig('plots/artificial_batch_size_fixed_update.png', bbox_inches='tight', dpi=200)
with open('plots/figures/artificial_batch_size_fixed_update.pkl', 'wb') as file:
    pickle.dump(ax, file)
plt.clf()
with open('plots/figures/artificial_batch_size_fixed_update.pkl', 'rb') as file:
    ax = pickle.load(file)
plt.title("Impact of artificially increasing batch size \n for fixed number of updates")






########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


ax = plt.subplot()
verbose = 3
time_limit = 10
update_limit = 3

kfac_params=[[0.18091478519897453,9543.235910696467,0.5279184525860896,0.6066011650262455]]

artificial_batch_sizes = [500, 1000, 2500, 5000, 7500, 10000, 15000, 20000]
batch_size = 100

colors = ["cornflowerblue", "indianred", "orangered", "seagreen", "palevioletred", "mediumpurple"]
for i, param in enumerate(kfac_params):
    losses = []
    lr, _, momen_rate, momen_max = param[0],int(param[1]),param[2],param[3]
    for artificial_batch_size in artificial_batch_sizes:
        net.initialize_weights_xavier()
        print("model ", i+1, "artificial_batch_size size: ", artificial_batch_size)
        loss, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                      artificial_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                      train_device, run_test, verbose, time_limit, update_limit)
        losses.append(t["inv_time"]/t["update_time"][1])


    plt.plot(artificial_batch_sizes, losses, color="cornflowerblue")

plt.legend()
plt.title("Inversion fraction for artificial batch sizes")
plt.xlabel("artificial batch size")
plt.ylabel("fraction of time")
plt.savefig('plots/artificial_batch_size_inversion_fraction.png', bbox_inches='tight', dpi=200)
with open('plots/artificial_batch_size_inversion_fraction.pkl','wb') as file:
    pickle.dump(ax, file)
# with open('plots/batch_size_fixed_updates.pkl', 'rb') as file:
#     ax = pickle.load(file)
plt.show()




kfac_params=[
        [0.18091478519897453,9543.235910696467,0.5279184525860896,0.6066011650262455],]
       #  [0.19801299042413312,10109.387282268859,0.7789782020549842,0.5446865337129331],
       #  [0.14375623156253733,4419.346043731717,1.1906234050193716,0.5560639333641116],
       #  [0.1594774014812677,4765.805432251613,1.762765115143839,0.3935509738186864]
       # ]

colors = ["cornflowerblue", "indianred", "orangered", "seagreen", "palevioletred", "mediumpurple", ""]
ax = plt.subplot()
verbose = 3
time_limit = 10
update_limit = 5000

artificial_batch_sizes = [25, 50, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 5000, ]

colors = ["cornflowerblue", "indianred", "orangered", "seagreen", "palevioletred", "mediumpurple",
          "goldenrod", "tomato", "chocolate", "powderblue"]
for i, param in enumerate(kfac_params):
    lr, _, momen_rate, momen_max = param[0],int(param[1]),param[2],param[3]
    c = -1
    while len(artificial_batch_sizes) > 1:
        c+=1
        losses = []
        batch_size = artificial_batch_sizes[0]
        for artificial_batch_size in artificial_batch_sizes:
            net.initialize_weights_xavier()
            print("model ", i+1, "artificial_batch_size size: ", artificial_batch_size)
            loss, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                          artificial_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                          train_device, run_test, verbose, time_limit, update_limit)
            losses.append(loss["test"][-1])

        plt.plot(artificial_batch_sizes, losses, color=colors[c])
        plt.scatter(artificial_batch_sizes[0], losses[0], color=colors[c], marker="o", label="B size: "+str(batch_size))
        plt.scatter(artificial_batch_sizes[np.argmin(np.asarray(losses))], min(losses), color=colors[c], marker="*")
        artificial_batch_sizes = artificial_batch_sizes[1:]

plt.legend()
plt.title("Impact of artificial batch size \n for gradually higher actual batch size")
plt.xlabel("artificial batch size")
plt.ylabel("test loss")
plt.ylim(0, 0.1)
plt.savefig('plots/artificial_batch_size_optimal.png', bbox_inches='tight', dpi=200)
with open('plots/figures/artificial_batch_size_optimal.pkl', 'wb') as file:
    pickle.dump(ax, file)
plt.show()
plt.clf()





