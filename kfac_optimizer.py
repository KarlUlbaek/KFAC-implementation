import time
from copy import deepcopy
import torch
from helper_functions_v2 import *
from main_functions import get_training_stats
from main_functions import print_stats





def kfac_optimizer(net, loss_func, problem_type, num_epochs, lr, num_layers, d, batch_size, desired_batch_size,
                   momen_max, momen_conv, momen_rate, scaling_1, scaling_2, train_device, run_test=False, verbose=1,
                   time_limit = 1E6, update_limit = 1E6, break_at_test_loss=False, l2_scaling = 0):
    # if non default limits are given epochs are set to a big number to not be the limit.
    if time_limit != 1E6 or update_limit != 1E6:
        num_epochs = int(1E6)

    net_params = list(net.parameters())
    A = {"ESTIMATE": [None for _ in range(num_layers)], "OUTER_PRODUCTS": [None for _ in range(num_layers)]}
    S, W, NG = deepcopy(A), deepcopy(A), deepcopy(A)
    W["GRAD_AVG"] = [None for _ in range(num_layers)]
    num_batches = d["train_size"] // batch_size
    fractions = [1E-6 for _ in range(num_layers)]  # needed to keep track of when the desired batch size is reached
    batch_ratio = batch_size / desired_batch_size # just a little safety proceadure when running optimization over hyper params
    update_count, batch_count = 0, 0
    losses = {"train":[], "test": [], "accuracy": []}
    test_loss_ = 1000000
    t = {"update_time": [], "inv_time": 0, "total_time":0}

    t["start"] = time.time()
    for epoch in range(num_epochs):
        # reshuffle the training data to get new batches
        random_perm = torch.randperm(d["train_size"])
        d["train"] = d["train"][random_perm]  # performing thic slice makes a copy
        if problem_type == 1 or problem_type == 2:
            d["train_targets"] = d["train_targets"][random_perm]
        else:
            d["train_targets"] = d["train"]  # therefor we must make a new reference in case of non classification

        for k in range(num_batches):

            # break when certain test lost is reached
            if break_at_test_loss:
                if break_at_test_loss > test_loss_:
                    return losses, t, net


            batch_count += 1

            # zeroth gradients but not by the usual optimizer.zero_grad() since we wont use an optimizer
            zeroth_gradients(net)

            # extract the next batch
            d["train_batch"] = d["train"][k * batch_size:k * batch_size + batch_size].to(train_device)
            if problem_type == 1 or problem_type == 2:
                d["train_target_batch"] = d["train_targets"][k * batch_size:k * batch_size + batch_size].to(train_device)
            else:
                d["train_target_batch"] = d["train_batch"].to(train_device)  # again we just make a reference to the training data


            # get the prediction as well as the s´s and the a´s
            train_prediction, S["VAL"], A["VAL"] = net.forward_s_and_a(d["train_batch"])
            if problem_type == 2: train_prediction = torch.squeeze(train_prediction)
            loss = loss_func(train_prediction, d["train_target_batch"])
            # regularization
            ############################################################################################################
            if l2_scaling > 0:
                l2_reg = torch.tensor(0., dtype=torch.float32, device=train_device)
                for param in net_params:
                    l2_reg += torch.norm(param)
                loss += l2_scaling * l2_reg
            ############################################################################################################
            loss.backward(retain_graph=True)


            # manually extract the weights and their gradients
            W["VAL"], W["GRAD"] = get_weights_and_gradients(net)

            # zeroth gradients again before getting the gradients of the networks target distribution
            zeroth_gradients(net)

            # enable gradients to be calculated on the s´s
            retain_grad(S["VAL"])

            # calculate gradients of the networks distribution by using random sampled predictions as if they
            # were the true labels
            if problem_type == 1:# for classification we need to take argmax to find the right label for cross entropy loss
                net_distribution = loss_func(train_prediction,
                                             torch.argmax(train_prediction, dim=1)[torch.randperm(batch_size)])
            else:
                net_distribution = loss_func(train_prediction, train_prediction[torch.randperm(batch_size)])

            net_distribution.backward()

            # collect the gradient
            S["GRAD"] = get_gradients(S["VAL"])

            with torch.no_grad():
                for i in range(num_layers):
                    if A["OUTER_PRODUCTS"][i] == None:
                        A["OUTER_PRODUCTS"][i] = batch_ratio * expectation_of_outer_products(A["VAL"][i])
                        S["OUTER_PRODUCTS"][i] = batch_ratio * expectation_of_outer_products(S["GRAD"][i])
                        W["GRAD_AVG"][i] = batch_ratio * W["GRAD"][i]
                        fractions[i] += batch_ratio

                    else:
                        A["OUTER_PRODUCTS"][i] += batch_ratio * expectation_of_outer_products(A["VAL"][i])
                        S["OUTER_PRODUCTS"][i] += batch_ratio * expectation_of_outer_products(S["GRAD"][i])
                        W["GRAD_AVG"][i] += batch_ratio * W["GRAD"][i]
                        fractions[i] += batch_ratio

                    # we only continue if the fraqtion has reached 1, meaning the outerproducts are done estimating
                    if fractions[i] > 1:
                        A["old_est"] = A["ESTIMATE"][i]
                        S["old_est"] = S["ESTIMATE"][i]
                        NG["old_est"] = NG["ESTIMATE"][i]
                        W["old_est"] = W["ESTIMATE"][i]  # most likely not used for anything

                        A["new_est"] = A["OUTER_PRODUCTS"][i]
                        S["new_est"] = S["OUTER_PRODUCTS"][i]
                        W["new_est"] = W["GRAD_AVG"][i]

                        # scale outer products and the gradients
                        A["new_est"] = rescale(A["new_est"], method=scaling_1)
                        S["new_est"] = rescale(S["new_est"], method=scaling_1)
                        W["new_est"] = rescale(W["new_est"], method=scaling_1)

                        # introduce momentum
                        A["new_est"] = momentum(A["new_est"], A["old_est"],
                                                max_=momen_max, k=update_count, rate=momen_rate, conv=momen_conv)
                        S["new_est"] = momentum(S["new_est"], S["old_est"],
                                                max_=momen_max, k=update_count, rate=momen_rate, conv=momen_conv)

                        # time the inverse time of the second inverse. we only do this once since we need to call
                        # cuda.synchronize to get a proper estimate and it slows down training
                        if update_count == 1:
                            torch.cuda.synchronize()
                            t["inv_time_temp"] = time.time()
                            A["inv"] = inv(A["new_est"], i, True, train_device)
                            S["inv"] = inv(S["new_est"])
                            torch.cuda.synchronize()
                            t["inv_time"] += time.time() - t["inv_time_temp"]
                        else:
                            A["inv"] = inv(A["new_est"], i, True, train_device)
                            S["inv"] = inv(S["new_est"])

                        # calc the natural gradient
                        NG["new_est"] = torch.mm(torch.mm(S["inv"], W["new_est"]), A["inv"])
                        # NG["raw"] = S["inv"] @ W["grad_scaled"] @ A["inv"]

                        # scale the natural gradient
                        NG["new_est"] = rescale(NG["new_est"], method=scaling_2)

                        # introduce momentum
                        NG["new_est"] = momentum(NG["new_est"], NG["old_est"],
                                                 max_=momen_max, k=update_count, rate=momen_rate, conv=momen_conv)

                        # update the weights according to the natural gradient and the lr
                        W_new = W["VAL"][i] - lr * NG["new_est"]

                        # update the weights in the network
                        net_params[i].copy_(W_new)
                        # net_param_i.copy_(W_new)

                        # Set it to None
                        A["OUTER_PRODUCTS"][i] = None
                        S["OUTER_PRODUCTS"][i] = None
                        W["GRAD_AVG"][i] = None

                        fractions[i] = 1E-6
                        A["ESTIMATE"][i] = A["new_est"]
                        S["ESTIMATE"][i] = S["new_est"]
                        NG["ESTIMATE"][i] = NG["new_est"]
                        W["ESTIMATE"][i] = W["new_est"]

                        # rest is just messy bookkeeping, deffo need to be cleaned up
                        ################################################################################################
                        ################################################################################################
                        ################################################################################################
                        ################################################################################################
                        ################################################################################################
                        # if we are at the final layer we can count the time used and increment the update counter
                        if i == num_layers - 1:
                            update_count += 1
                            t["update_time"].append(time.time() - t["start"])
                            t["total_time"] += t["update_time"][-1]
                            # reset timer
                            t["start"] = time.time()

                            # get stats and print every update
                            if verbose == 1:
                                train_loss_, test_loss_, accuracy_, current_momentum = \
                                    get_training_stats(momen_max, update_count, momen_rate, momen_conv, net, d,
                                                       problem_type, loss_func, run_test, train_device)

                                losses["train"].append(train_loss_)
                                losses["test"].append(test_loss_)
                                losses["accuracy"].append(accuracy_)
                                print_stats(epoch, batch_count, update_count, train_loss_, test_loss_, accuracy_,
                                            current_momentum, lr, t)

                                # reset the timer every time we get the stats since we dont wonna track of how long that takes
                                t["start"] = time.time()

                            # return becuase time or update limit is reached
                            if t["total_time"] + t["update_time"][-1]*0.5 > time_limit or update_count >= update_limit:
                                if verbose != 1:
                                    # get stats if we are about to return because of time or update limit and havnt just gotten
                                    # stats from verbose == 1
                                    train_loss_, test_loss_, accuracy_, current_momentum = \
                                        get_training_stats(momen_max, update_count, momen_rate, momen_conv, net, d,
                                                           problem_type, loss_func, run_test, train_device)

                                    losses["train"].append(train_loss_)
                                    losses["test"].append(test_loss_)
                                    losses["accuracy"].append(accuracy_)
                                    print_stats(epoch, batch_count, update_count, train_loss_, test_loss_, accuracy_,
                                                current_momentum, lr, t)

                                return losses, t, net

            # get stats and print every epoch
        if verbose == 2:
            # only get the new stats every epoch since it takes some time.
            train_loss_, test_loss_, accuracy_, current_momentum = \
                get_training_stats(momen_max, update_count, momen_rate, momen_conv, net, d,
                                   problem_type, loss_func, run_test, train_device)

            losses["train"].append(train_loss_)
            losses["test"].append(test_loss_)
            losses["accuracy"].append(accuracy_)
            print_stats(epoch, batch_count, update_count, train_loss_, test_loss_, accuracy_,
                        current_momentum, lr, t)

            # reset the timer every time we get the stats since we dont wonna track how long that takes
            t["start"] = time.time()

        # get final stats if we dont already have done so
    if verbose != 1 and verbose != 2:
        train_loss_, test_loss_, accuracy_, current_momentum = \
            get_training_stats(momen_max, update_count, momen_rate, momen_conv, net, d,
                               problem_type, loss_func, run_test, train_device)

        losses["train"].append(train_loss_)
        losses["test"].append(test_loss_)
        losses["accuracy"].append(accuracy_)

        # print before returning
    if verbose == 3:
        print_stats(epoch, batch_count, update_count, train_loss_, test_loss_, accuracy_,
                    current_momentum, lr, t)

    return losses, t, net

