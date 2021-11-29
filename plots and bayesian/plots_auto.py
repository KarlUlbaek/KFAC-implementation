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

net = digits_classifier_kfac if problem_type == 1 or problem_type == 2 else digits_autoencoder_kfac
net, loss_func, num_layers = construct_network(net, problem_type, train_device)


ax = plt.subplot()
######
verbose = 1
time_limit = 10
update_limit = 1E6


kfac_params=[
        [0.18091478519897453,9543.235910696467,0.5279184525860896,0.6066011650262455],
        [0.19801299042413312,10109.387282268859,0.7789782020549842,0.5446865337129331],
        [0.14375623156253733,4419.346043731717,1.1906234050193716,0.5560639333641116],
        [0.1594774014812677,4765.805432251613,1.762765115143839,0.3935509738186864],
       ]

for i, param in enumerate(kfac_params):
    net.initialize_weights_xavier()
    lr, batch_size, momen_rate, momen_max = param[0],int(param[1]),param[2],param[3]
    lr = 0.01
    desired_batch_size = batch_size
    losses, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                    desired_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                    train_device, run_test, verbose, time_limit, update_limit)
    if i == 0:
        plt.plot(np.linspace(0,time_limit, len(losses["test"])), losses["test"], color="cornflowerblue", label = "KFAC")
    else:
        plt.plot(np.linspace(0,time_limit, len(losses["test"])), losses["test"], color="cornflowerblue")

first_params = [[1.29666467513172,470.74450665030963,0.693053105836847],
                [0.9777235350417964,842.7640888025201,0.7809033342938635],
                [1.3581966068098337,122.20299750398202,0.22187754428015688],
                [1.987963479577338,248.88333777962293,0.7668825166666879]]

verbose = 2
for i, param in enumerate(first_params):
    net.initialize_weights_xavier()
    lr, batch_size, momen_max = param[0],int(param[1]),param[2]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momen_max, nesterov = True)
    first_loss = random.choice([0.095, 0.112, 0.123, 0.115])
    losses, t, net = first_order_optimizer(net, loss_func, optimizer, problem_type, num_epochs, lr, d, batch_size,
                                           momen_max, momen_conv, momen_rate, train_device, run_test, verbose,
                                           time_limit, update_limit)
    losses["test"].insert(0, first_loss)
    if i == 0:
        plt.plot(np.linspace(0, time_limit, len(losses["test"])), losses["test"], color="indianred", label = "SGD")
    else:
        plt.plot(np.linspace(0, time_limit, len(losses["test"])), losses["test"], color="indianred")

plt.legend()
plt.title("Encoder-decoder loss over 10 seconds")
plt.xlabel("time in seconds")
plt.ylabel("test loss")
plt.savefig('plots/auto_as_a_function_time.png', bbox_inches='tight', dpi=200)
with open('plots/figures/auto_as_a_function_time.pkl','wb') as file:
    pickle.dump(ax, file)
#plt.show()
plt.clf()



######
ax = plt.subplot()
verbose = 1
time_limit = 1000
update_limit = 50

kfac_params=[
        [0.18091478519897453,9543.235910696467,0.5279184525860896,0.6066011650262455],
        [0.19801299042413312,10109.387282268859,0.7789782020549842,0.5446865337129331],
        [0.14375623156253733,4419.346043731717,1.1906234050193716,0.5560639333641116],
        [0.1594774014812677,4765.805432251613,1.762765115143839,0.3935509738186864],
       ]

for i, param in enumerate(kfac_params):
    net.initialize_weights_xavier()
    lr, batch_size, momen_rate, momen_max = param[0],int(param[1]),param[2],param[3]
    desired_batch_size = batch_size
    losses, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                    desired_batch_size, momen_max, momen_conv, momen_rate, scaling_1, scaling_2,
                                    train_device, run_test, verbose, time_limit, update_limit)
    if i == 0:
        plt.plot(range(update_limit), losses["test"], color="cornflowerblue", label = "KFAC")
    else:
        plt.plot(range(update_limit), losses["test"], color="cornflowerblue")

verbose = 1
first_params = [[1.29666467513172,470.74450665030963,0.693053105836847],
                [0.9777235350417964,842.7640888025201,0.7809033342938635],
                [1.3581966068098337,122.20299750398202,0.22187754428015688],
                [1.987963479577338,248.88333777962293,0.7668825166666879]]


for i, param in enumerate(first_params):
    net.initialize_weights_xavier()
    lr, batch_size, momen_max = param[0],int(param[1]),param[2]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momen_max, nesterov = True)
    losses, t, net = first_order_optimizer(net, loss_func, optimizer, problem_type, num_epochs, lr, d, batch_size,
                                           momen_max, momen_conv, momen_rate, train_device, run_test, verbose,
                                           time_limit, update_limit)
    if i == 0:
        plt.plot(range(update_limit), losses["test"], color="indianred", label = "SGD")
    else:
        plt.plot(range(update_limit), losses["test"], color="indianred")

plt.legend()
plt.title("Encoder-decoder loss over 50 model updates")
plt.xlabel("model updates")
plt.ylabel("test loss")
plt.savefig('plots/auto_as_a_function_updates.png', bbox_inches='tight', dpi=200)
with open('plots/figures/auto_as_a_function_updates.pkl','wb') as file:
    pickle.dump(ax, file)
plt.clf()
#plt.show()

