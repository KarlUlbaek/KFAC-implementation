from main_functions import *
from network_architectures import digits_autoencoder_kfac, digits_classifier_kfac
from kfac_optimizer import kfac_optimizer
import pandas as pd


#data_storage_device = "cuda:0"
data_storage_device = "cpu"
# train_device = "cpu"
train_device = "cuda:0"


# data dict
train_data_location = "data/cifar_rgb_train.npy"
train_target_location = "data/cifar_rgb_train_targets.npy"
test_data_location = "data/cifar_rgb_test.npy"
test_target_location = "data/cifar_rgb_test_targets.npy"

# hyper params
num_epochs = 15
batch_size = 20000
desired_batch_size = 20000
update_limit = 1E6

lr = 0.2
scaling_1 = 3 # 3 is prolly the best option since only 3 works for classification
scaling_2 = 3 # 3 is the only viable option
momen_max = 0.90  # if set to 0 there will be no momentum
momen_rate = 0.25  # between 0 and 1, the greater the value the faster the max momentum will be reached.
momen_conv = "linear"#"exp"  # "linear, "exp", "constant", "paper". It determines how the momentum will converve uopon start up



problem_type = 1
run_test = True
verbose = 1
time_limit = 10
repetitions = 2
initial_points = 20
max_iter = 10000


d = load_data(train_data_location, train_target_location, batch_size, data_storage_device,
              run_test, problem_type, test_data_location, test_target_location)

net = digits_classifier_kfac if problem_type == 1 or problem_type == 2 else digits_autoencoder_kfac
net, loss_func, num_layers = construct_network(net, problem_type, train_device,
                                               net_struct = [3000, 2000, 1500, 1000 ,250, 10],
                                               input_size=int(d["num_attributes"]))

def baysian_black_box(to_optimize, net, problem_type, num_epochs, d, momen_conv, scaling_1, scaling_2, train_device,
                      run_test, verbose, loss_func, num_layers, time_limit):


    # lr, batch_size, desired_batch_size, momen_rate, momen_max = to_optimize[0,0], to_optimize[0,1], to_optimize[0,2], \
    #                                                             to_optimize[0,3], to_optimize[0,4]
    lr, batch_size, momen_rate, momen_max = to_optimize[0,0], to_optimize[0,1], to_optimize[0,2], to_optimize[0,3]
    batch_size = int(batch_size)
    desired_batch_size = batch_size

    print("lr=", lr, "batch_size=", batch_size, "desired_batch_size=", desired_batch_size, "momen_rate=", momen_rate,
          "momen_max=", momen_max)

    losses = 0
    for i in range(repetitions):
        print("Repetition nr", i+1)
        loss, t, net = kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size,
                                      desired_batch_size, momen_max, momen_conv, momen_rate, scaling_1,
                                      scaling_2, train_device, run_test, verbose, time_limit=time_limit)

        net.initialize_weights_xavier() #reset the weights inbetween runs
        losses += min(loss["test"]) # append losses

    return losses / repetitions


from functools import partial
baysian_black_box = partial(baysian_black_box, net=net, problem_type=problem_type,
                            num_epochs=num_epochs, d=d, momen_conv=momen_conv, scaling_1=scaling_1, scaling_2=scaling_2,
                            train_device=train_device, run_test=run_test, verbose=verbose, loss_func=loss_func,
                            num_layers=num_layers, time_limit=time_limit)

# 0.010641762055456638 [1.14321587e-01 1.85703482e+03 4.52024016e+03 1.06963152e+00 4.25890005e-01]
from GPyOpt.methods import BayesianOptimization
mixed_domain =[
                {'name': 'lr', 'type': 'continuous', 'domain': (0.01, 0.5)},
                {'name': 'batch_size', 'type': 'continuous', 'domain': (500, 10000)},
                #{'name': 'desired_batch_size', 'type': 'continuous', 'domain': (2000, 60000)},
                {'name': 'momen_rate', 'type': 'continuous', 'domain': (0.01, 2)},
                {'name': 'momen_max', 'type': 'continuous', 'domain': (0, 0.95)}
              ]

#constraints = [{'name' : 'batch_size_cons1','constraint' : 'x[:,1] - x[:,2]'},
#               {'name' : 'batch_size_cons2','constraint' : 'np.ceil(x[:,2] / x[:,1])*x[:,1]-60001'}]

opt = BayesianOptimization(f=baysian_black_box, initial_design_numdata = initial_points, domain=mixed_domain)#, constraints=constraints)

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
        df["momen_rate"] = opt.X[:,2]
        df["momen_max"] = opt.X[:,3]
        df = df.sort_values(by=['loss'])
        print(df.head(5))
        #df.to_csv(input("name the csv you fucking donky!")+".csv", index=False)
        df.to_csv("kfac_class_v1.csv", index=False)

        # only increment if the loop was succesfull
        i+=1

    except Exception:
        print("exception occured lets hope it wont happen again : )")



# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats(10)

# if not classification: display(d["train"], net.forward(d["train"]), num_examples=10, close=False)
# plt.plot(losses["train"])
# plt.show()




