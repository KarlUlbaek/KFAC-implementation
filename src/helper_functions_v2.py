from torch import nn
import torch
import numpy as np

def one_hot(a):
    a = a.astype(int)
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b

def adaptive_learning_rate(lr_cur, losses, method):
    if method == 0: # constant
        return lr_cur

    if method == 1: # manual scheduing:
        loss = losses[-1]
        schedule = [(1,0.2), (0.02,0.15), (0.015,0.1), (0.01,0.05)]
        for loss_s, lr in schedule:
            if loss < loss_s: return lr

        return lr_cur

    if method == 2: # propertinal to loss:
        loss = losses[-1]
        method_2_scaling = 10
        method_2_max = 0.2
        return min(method_2_max, loss*method_2_scaling)

    if method == 3: # average adaptive:
        if len(losses) < 2:
            return lr_cur

        loss = losses[-1]
        last_x_itr = 5
        tolerance = 1.02
        inc, decre = 1.02, 0.85
        if loss < tolerance*np.mean(np.array(losses[-(last_x_itr+1):-1])):
            return lr_cur*inc
        else:
            return lr_cur*decre



def formatted_matrix_stats(M, M_old = None, L ="A", diff_method ="percent"):
    if not isinstance(M, list):
        M = [M]

    M_format = ""
    for i in range(len(M)):
        mean = torch.mean(M[i]).cpu().item()
        std = torch.std(M[i]).cpu().item()

        if mean < 0: # since it can be negative and we must give it more space
            M_format += "{}{}:({:.2e}, {:.2e}) ".format(L, i, mean, std)
        else:
            M_format += "{}{}:( {:.2e}, {:.2e}) ".format(L, i, mean, std)

        if M_old is not None:
            if diff_method == "percent":
                diff = torch.sum(torch.abs(M[i]))
                #diff = torch.mean(torch.abs(torch.div((M_old[i] - M[i]), M[i])))

            else:
                diff = torch.sum(torch.abs(M[i]))
                #diff = torch.sum(torch.abs(M[i]-M_old[i]))

            diff = diff if diff==diff else 0 # check if its nan
            M_format = M_format[:-2]
            if diff < 1 and diff != 0:
                M_format += ", {:#.2g}) ".format(diff)
            else:
                M_format += ", {:#.3g}) ".format(diff)


    return M_format

def momentum(M_new, M, max_ = 0.9, k = None, rate = 0.5, conv ="linear"):
    if M == None or max_ == 0:
        return M_new

    # max momentum from the start
    elif conv == "constant":
        momentum = max_
        return momentum*M + (1-momentum)*M_new

    # Linearly increasing the momentum. takes roughly 50 batches to reach full effect
    elif conv == "linear":
        momentum = min(0.05*k*rate, max_)
        return momentum * M + (1 - momentum) * M_new

    # exponentially increasing the momentum. takes roughly 50 batches to reach full effect
    elif conv == "exp":
        momentum = min(1-np.exp(-0.1*k*rate), max_)
        return momentum * M + (1 - momentum) * M_new

    # about the same as the one from k-fac paper. takes roughly 50 batches to reach full effect
    elif conv == "paper":
        momentum = max(min(1-1/(k*rate*0.4), max_), 0.15)
        return momentum * M + (1 - momentum) * M_new

    else:
        print("please select a momentum convergence type/method")

def get_gradients(list_of_weights):
    return [w.grad for w in list_of_weights]

def retain_grad(list_of_weights):
    for w in list_of_weights: w.retain_grad()


def expectation_of_outer_products(a):
    return torch.mm(a.T, a) / a.shape[0]

def inv(M, idx = 1, is_A = False, device="cuda:0"):
    if idx == 0 and is_A:
        return torch.inverse(M+torch.rand(M.shape, device=device)*0.0000001)
    else:
        try:
            return torch.inverse(M)
        except RuntimeError:
            print("Inversion error ourred. Noise was added.")
            return torch.inverse(M + torch.rand(M.shape, device=device) * 0.0000001)



def rescale(M1, M2=None, method=0):
    if method == 0:
        return M1

    # methods:
    # according to M2
    # 1: rescale according to the other matrix
    if M2 is not None:
        # scale to match mean
        if method == 1:
            current_mean = torch.mean(M1)
            desired_mean = torch.mean(M2)
            return M1 * (desired_mean / current_mean)

        # shift to match mean
        if method == 2:
            current_mean = torch.mean(M1)
            desired_mean = torch.mean(M2)
            return M1 - (current_mean-desired_mean)

        # scale to match max
        current_max = torch.max(M1)
        desired_max = torch.max(M2)
        return M1 * (desired_max / current_max)

    else:
        # according to itself
        # 1: 0 mean
        if method == 1:
            return M1 - torch.mean(M1)

        # 2: 0 mean and std 1
        if method == 2:
            return (M1 - torch.mean(M1)) / torch.std(M1)

        # default: normalize such that the biggest value becomes 1:
        if method == 3:
            return M1 / torch.max(M1)

        if method == 4: #scaled by square root of the mean
            return (torch.abs(torch.mean(M1))**(1/2))*(M1 / torch.max(M1))

        if method == 5: #scaled by the 4. root of the mean
            return (torch.abs(torch.mean(M1))**(1/4))*(M1 / torch.max(M1))

        if method == 6: #normalize to vector with length one
            length = torch.sqrt(torch.sum(M1**2))
            return M1 / length

        if method == 7: #normalize to vector with length one and then scaled with sqrt(num parameters)
            x_dim, y_dim = M1.shape
            length = torch.sqrt(torch.sum(M1**2))
            return M1 / length * np.sqrt(x_dim*y_dim)

        # scale to be between -1 and 1
        if method == 8:
            range_upper = 1
            range_lower = -1
            m_min = M1.min()
            m_max = M1.max()

            return ((M1 - m_min) / (m_max - m_min)) * (range_upper-range_lower) - range_upper



def get_weights_and_gradients(net):
    weights = []
    weight_gradients = []
    for w in net.parameters():
        weights.append(torch.clone(w))
        weight_gradients.append(torch.clone(w.grad))

    return weights, weight_gradients

def zeroth_gradients(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.Module.zero_grad(m)

def calc_pi(A, S_g):
    A_d = A.shape[0]
    S_g_d = S_g.shape[0]

    pi = torch.sqrt((torch.trace(A)/(A_d+1)) / (torch.trace(S_g)/S_g_d))

    return pi

def lambda_regularization(M, lamb, pi, device):
    return M + pi*lamb*torch.eye(M.shape[0], device=device)



