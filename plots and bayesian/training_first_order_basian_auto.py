#from kfac_optimizer import kfac_optimizer
from main_functions import *
import cProfile
import pstats
import matplotlib.pyplot as plt
from display_func import display
from network_architectures import digits_autoencoder_kfac, digits_classifier_kfac
from first_order_optimizer import first_order_optimizer
import pandas as pd

# import random
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

# data_storage_device = "cuda:0"
data_storage_device = "cpu"
# train_device = "cpu"
train_device = "cuda:0"


# data dict
train_data_location = "data/digits_train.npy"
train_target_location = "data/digits_train_targets.npy"
test_data_location = "data/digits_test.npy"
test_target_location = "data/digits_test_targets.npy"

# hyper params
num_epochs = 10
batch_size = 20000
desired_batch_size = 20000
update_limit = 1E6

lr = 0.05
momen_max = 0.9  # if set to 0 there will be no momentum
momen_rate = None  # only needed for kfac but is needed as an argument in order to use the same functions
momen_conv = "constant" # "linear, "exp", "constant", "paper". It determines how the momentum will converve uopon start up


problem_type = 0
run_test = True
verbose = 3
time_limit = 10
repetitions = 2
initial_points = 20
max_iter = 10000

d = load_data(train_data_location, train_target_location, batch_size, data_storage_device,
              run_test, problem_type, test_data_location, test_target_location)

net = digits_classifier_kfac if problem_type==1 or problem_type==2 else digits_autoencoder_kfac
net, loss_func, num_layers = construct_network(net, problem_type, train_device)

def baysian_black_box(to_optimize, net, loss_func, problem_type, num_epochs, d, momen_conv, momen_rate,
                      train_device, run_test, verbose, time_limit, update_limit):

    lr, batch_size, momen_max = to_optimize[0,0], to_optimize[0,1], to_optimize[0,2]
    batch_size = int(batch_size)

    print("\nlr=", lr, "batch_size=", batch_size, "momen_max=", momen_max)

    losses = 0
    for i in range(repetitions):
        print("Repetition nr", i+1)
        net.initialize_weights_xavier() #reset the weights inbetween runs
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momen_max, nesterov = True) # reset the optimizer
        loss, t, net = first_order_optimizer(net, loss_func, optimizer, problem_type, num_epochs, lr, d, batch_size,
                                             momen_max, momen_conv, momen_rate, train_device, run_test, verbose,
                                             time_limit, update_limit)

        losses += loss["test"][-1] # append losses

    return losses / repetitions


from functools import partial
baysian_black_box = partial(baysian_black_box, net=net, loss_func=loss_func,
                            problem_type=problem_type, num_epochs=num_epochs, d=d, momen_conv=momen_conv,
                            momen_rate=momen_rate, train_device=train_device, run_test=run_test, verbose=verbose,
                            time_limit=time_limit, update_limit=update_limit)

from GPyOpt.methods import BayesianOptimization
mixed_domain =[
                {'name': 'lr', 'type': 'continuous', 'domain': (0.01, 2)},
                {'name': 'batch_size', 'type': 'continuous', 'domain': (10, 5000)},
                {'name': 'momen_max', 'type': 'continuous', 'domain': (0, 0.99)}
              ]
#
# import cProfile
# import pstats
# with cProfile.Profile() as pr:
opt = BayesianOptimization(f=baysian_black_box, initial_design_numdata = initial_points, domain=mixed_domain)#, constraints=constraints)
# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats(10)

i = 0
while i < 3:
    # will run for half an hour each time and save current progress after each half hour. so 8 hours total
    print("running on qaurter:", i)
    try:
        opt.run_optimization(max_iter=max_iter, max_time=900, verbosity = True)


        # print(opt.fx_opt, opt.x_opt)
        # print(opt.X, opt.Y)

        df = pd.DataFrame(opt.Y, columns=['loss'])
        df["lr"] = opt.X[:,0]
        df["batch_size"] = opt.X[:,1]
        #df["desired_batch_size"] = opt.X[:,2]
        df["momen_max"] = opt.X[:,2]
        df = df.sort_values(by=['loss'])
        print(df.head(5))
        #df.to_csv(input("name the csv you fucking donky!")+".csv", index=False)
        df.to_csv("1.order_auto_v2.csv", index=False)

        # only increment if the loop was succesfull
        i+=1

    except Exception:
        print("exception occured lets hope it wont happen again : )")




