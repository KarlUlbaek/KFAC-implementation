import time
from helper_functions_v2 import *
from main_functions import get_training_stats
from main_functions import print_stats

def first_order_optimizer(net, loss_func, optimizer, problem_type, num_epochs, lr, d, batch_size, momen_max,
                          momen_conv, momen_rate=None, train_device="cuda:0", run_test=False, verbose=1,
                          time_limit = 1E6, update_limit = 1E6, break_at_test_loss=False):

    # if non default limits are given epochs are set to a big number to not be the limit.
    if time_limit != 1E6 or update_limit != 1E6:
        num_epochs = int(1E6)

    num_batches = d["train_size"] // batch_size
    update_count, batch_count = 0, 0
    losses = {"train":[], "test": [], "accuracy": []}
    test_loss_ = 1E6
    t = {"update_time": [], "inv_time": 0, "total_time":0}

    t["start"] = time.time()
    for epoch in range(num_epochs):
        # reshuffle the training data to get new batches
        random_perm = torch.randperm(d["train_size"])
        d["train"] = d["train"][random_perm]  # performing this slice makes a copy
        if problem_type == 1 or problem_type == 2:
            d["train_targets"] = d["train_targets"][random_perm]
        else:
            d["train_targets"] = d["train"]  # therefore we must make a new reference in case of non classification

        for k in range(num_batches):
            # break when certain test lost is reached
            if break_at_test_loss:
                if break_at_test_loss > test_loss_:
                    return losses, t, net


            batch_count += 1

            # extract the next batch
            d["train_batch"] = d["train"][k * batch_size:k * batch_size + batch_size].to(train_device)
            if problem_type == 1 or problem_type == 2:
                d["train_target_batch"] = d["train_targets"][k * batch_size:k * batch_size + batch_size].to(train_device)
            else:
                d["train_target_batch"] = d["train_batch"].to(train_device)  # again we just make a reference to the training data


            # this is basically the whole thing
            optimizer.zero_grad()
            train_prediction = net.forward(d["train_batch"])
            if problem_type == 2: train_prediction = torch.squeeze(train_prediction)
            loss = loss_func(train_prediction, d["train_target_batch"])
            loss.backward()
            optimizer.step()

            # rest is just messy bookkeeping
            ###########################################################################################################
            ###########################################################################################################
            ###########################################################################################################
            ###########################################################################################################
            update_count += 1
            t["update_time"].append(time.time() - t["start"])
            t["total_time"]+= t["update_time"][-1]
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
            if t["total_time"]+t["update_time"][-1] > time_limit or update_count >= update_limit:
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
            train_loss_, test_loss_, accuracy_, current_momentum= \
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


