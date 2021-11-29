from helper_functions_v2 import momentum
import numpy as np
import torch
from torch import nn


# digits_classifier_kfac
# digits_autoencoder_kfac
def construct_network(net, problem_type, train_device, net_struct=None, activation_func=nn.Tanh(), net_work_scaling=1,
                      input_size=784, bias=False):
    if problem_type == 0:
        if net_struct is None: net_struct = [1000, 500, 250, 30]
        num_layers = len(net_struct) * 2
        net_struct = [int(s * net_work_scaling) for s in net_struct]
        net = net(structure=net_struct, activation_func=activation_func, input_size=input_size, bias=bias).to(
            device=train_device)
        loss_func = nn.MSELoss()

    elif problem_type == 1:
        if net_struct is None: net_struct = [1000, 750, 500, 250, 100, 10]
        num_layers = len(net_struct)
        net_struct = [int(s * net_work_scaling) for s in net_struct[:-1]] + [net_struct[-1]]
        net = net(structure=net_struct, activation_func=activation_func, input_size=input_size, bias=bias).to(
            device=train_device)
        loss_func = nn.CrossEntropyLoss()

    else:
        if net_struct is None: net_struct = [500, 750, 500, 250, 100, 1]
        num_layers = len(net_struct)
        net_struct = [int(s * net_work_scaling) for s in net_struct[:-1]] + [net_struct[-1]]
        net = net(structure=net_struct, activation_func=activation_func, input_size=input_size, bias=bias).to(
            device=train_device)
        loss_func = nn.MSELoss()

    net.initialize_weights_xavier()


    return net, loss_func, num_layers


def print_stats(epoch, batch_count, update_count, train_loss, test_loss, accuracy, current_momentum, lr, t):
    if accuracy == None:
        accuracy = ""
    else:
        accuracy = "Acc="+(str(accuracy)+"    ")[:4]+" "

    if test_loss == None:
        test_loss = ""
    else:
        test_loss = "Test="+(str(test_loss)+"    ")[:6]+" "

    print(("#Epoch {:3s} #batch {:3s} #update {:3s} Train={:0.4f} {:s}{:s}Momen={:0.3f} LR={:0.3f} "
           "update_time={:0.3f} inv_time={:0.3f} total_time={:0.3f}".format(str(epoch + 1),
                                                         str(batch_count),
                                                         str(update_count),
                                                         train_loss,
                                                         test_loss,
                                                         accuracy,
                                                         current_momentum,
                                                         lr,
                                                         t["update_time"][-1],
                                                         t["inv_time"],
                                                         t["total_time"])
                                                         ))

def get_training_stats(momen_max, update_count, momen_rate, momen_conv, net, d, problem_type,
                       loss_func, run_test, train_device):
    with torch.no_grad():
        # lr = adaptive_learning_rate(lr, losses, method=0)
        current_momentum = momentum(0, 1, max_=momen_max, k=update_count, rate=momen_rate, conv=momen_conv)

        # make a farward pass of the first x images in the train set, where x is the size of the test set
        train_subset = (d["train"][: d["test_size"]]).to(train_device)
        train_targets_subset = (d["train_targets"][: d["test_size"]]).to(train_device)


        train_prediction_this_epoch = net.forward(train_subset)
        if problem_type == 2: train_prediction_this_epoch = torch.squeeze(train_prediction_this_epoch)
        accuracy, test_loss = None, None
        # if classification:
        #     accuracy = torch.argmax(train_prediction_this_epoch, dim=1)
        #     accuracy = (accuracy == train_targets_subset).sum().item() / d["test_size"]
        # else:
        #     # for autoencoder we calc mean abs error
        #     accuracy = torch.mean(torch.abs(train_prediction_this_epoch - train_targets_subset)).item()
        #     # accuracy = nn.L1Loss(train_prediction_this_epoch, (d["train_targets"][: d["test_size"]]))

        train_loss = loss_func(train_prediction_this_epoch, train_targets_subset).item()
        # print either train losses or train and test losses
        if run_test:
            test_prediction = net.forward(d["test"].to(train_device))
            if problem_type == 2: test_prediction = torch.squeeze(test_prediction)
            test_loss = loss_func(test_prediction, d["test_targets"].to(train_device)).item()
            if problem_type == 1:
                accuracy = torch.argmax(net.forward(d["test"].to(train_device)), dim=1)
                accuracy = (accuracy == d["test_targets"].to(train_device)).sum().item() / d["test_size"]

        return train_loss, test_loss, accuracy, current_momentum


def load_data(train_data_location, train_target_location, batch_size, data_storage_device, run_test=False,
              problem_type=False, test_data_location=None, test_target_location=None, scaling = 255):
    # d for data ofc
    d = {}
    d["train"] = np.load(train_data_location) / scaling
    d["num_attributes"] = d["train"].shape[1]
    d["train"] = torch.from_numpy(d["train"]).type(torch.float32).to(device=data_storage_device)
    d["train_targets"] = d["train"]  # just a reference to the same object

    if run_test:
        d["test"] = np.load(test_data_location) / scaling
        d["test"] = torch.from_numpy(d["test"]).type(torch.float32).to(device=data_storage_device)
        d["test_targets"] = d["test"]  # just a reference to the same object
        d["test_size"] = d["test"].shape[0]

    if problem_type == 1:
        d["train_targets"] = np.load(train_target_location)
        # d["train_targets"] = one_hot(d["train_targets"])
        d["train_targets"] = torch.from_numpy(d["train_targets"]).type(torch.long).to(device=data_storage_device)

        if run_test:
            d["test_targets"] = np.load(test_target_location)
            # d["test_targets"] = one_hot(d["test_targets"])
            d["test_targets"] = torch.from_numpy(d["test_targets"]).type(torch.long).to(device=data_storage_device)

    if problem_type == 2:
        d["train_targets"] = np.load(train_target_location)
        # d["train_targets"] = one_hot(d["train_targets"])
        d["train_targets"] = torch.from_numpy(d["train_targets"]).type(torch.float32).to(device=data_storage_device)

        if run_test:
            d["test_targets"] = np.load(test_target_location)
            # d["test_targets"] = one_hot(d["test_targets"])
            d["test_targets"] = torch.from_numpy(d["test_targets"]).type(torch.float32).to(device=data_storage_device)

    if not run_test: d["test_size"] = batch_size
    d["train_size"] = d["train"].shape[0]
    return d