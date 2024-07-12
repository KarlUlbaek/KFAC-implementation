import torch.nn as nn
import torch

class kfac_dummy_network(nn.Module):
    def __init__(self, input_size = 4, latent_size = 3, activation_func = nn.ReLU(), bias = False):
        super(kfac_dummy_network, self).__init__()
        self.activation = activation_func
        self.input_size = input_size
        self.latent_size = latent_size

        self.encode = nn.Linear(input_size, latent_size, bias=bias)
        self.decode = nn.Linear(latent_size, input_size, bias=bias)

    def forward_s_and_a(self, a0):
        s1 = self.encode(a0)
        a1 = self.activation(s1)
        s2 = self.decode(a1)
        a2 = self.activation(s2)
        return a2, [s1, s2], [a0, a1]

    def forward(self, x):
        x1 = self.activation(self.encode(x))
        x2 = self.activation(self.decode(x1))
        return x2

    def initialize_weights_xavier(self, bias = 0.5):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, bias)


class digits_autoencoder_kfac(nn.Module):
    def __init__(self, structure = [1000, 500, 250, 30], input_size=784, activation_func = nn.Tanh(), bias = False):
        super(digits_autoencoder_kfac, self).__init__()
        self.activation = activation_func
        self.s = structure

        # Inputs to hidden layer linear transformation
        self.encode1 = nn.Linear(input_size, self.s[0], bias=bias)
        self.encode2 = nn.Linear(self.s[0],  self.s[1], bias=bias)
        self.encode3 = nn.Linear(self.s[1],  self.s[2], bias=bias)
        self.encode4 = nn.Linear(self.s[2],  self.s[3], bias=bias)

        self.decode5 = nn.Linear(self.s[3],  self.s[2], bias=bias)
        self.decode6 = nn.Linear(self.s[2],  self.s[1], bias=bias)
        self.decode7 = nn.Linear(self.s[1],  self.s[0], bias=bias)
        self.decode8 = nn.Linear(self.s[0],  input_size, bias=bias)

    def forward_s_and_a(self, a0):
        # encoding:
        s1 = self.encode1(a0)
        a1 = self.activation(s1)
        s2 = self.encode2(a1)
        a2 = self.activation(s2)
        s3 = self.encode3(a2)
        a3 = self.activation(s3)
        s4 = self.encode4(a3)
        a4 = self.activation(s4)

        # decoding
        s5 = self.decode5(a4)
        a5 = self.activation(s5)
        s6 = self.decode6(a5)
        a6 = self.activation(s6)
        s7 = self.decode7(a6)
        a7 = self.activation(s7)
        s8 = self.decode8(a7)
        #a8 = self.activation(s8)

        # WITHOUT ACTIVATION
        return s8, [s1, s2, s3, s4, s5, s6, s7, s8], [a0, a1, a2, a3, a4, a5, a6, a7]

    def forward(self, x):
        x1 = self.activation(self.encode1(x))
        x2 = self.activation(self.encode2(x1))
        x3 = self.activation(self.encode3(x2))
        x4 = self.activation(self.encode4(x3))
        x5 = self.activation(self.decode5(x4))
        x6 = self.activation(self.decode6(x5))
        x7 = self.activation(self.decode7(x6))
        return self.decode8(x7)
        #return self.activation(self.decode8(x7))


    def initialize_weights_xavier(self, bias = 0.5):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, bias)



class digits_classifier_kfac(nn.Module):
    def __init__(self, structure = [1000, 750, 500, 250 ,100, 10], input_size=784, activation_func = nn.Tanh(), bias = False):
        super(digits_classifier_kfac, self).__init__()
        self.activation = activation_func
        self.s = structure


        self.fully_connected1 = nn.Linear(input_size, self.s[0], bias=bias)
        self.fully_connected2 = nn.Linear(self.s[0],  self.s[1], bias=bias)
        self.fully_connected3 = nn.Linear(self.s[1],  self.s[2], bias=bias)
        self.fully_connected4 = nn.Linear(self.s[2],  self.s[3], bias=bias)
        self.fully_connected5 = nn.Linear(self.s[3],  self.s[4], bias=bias)
        self.fully_connected6 = nn.Linear(self.s[4],  self.s[5], bias=bias)


    # def forward_s_and_a(self, a0):
    #     # encoding:
    #     s1 = self.fully_connected1(a0)
    #     a1 = self.activation(s1)
    #     s2 = self.fully_connected2(a1)
    #     a2 = self.activation(s2)
    #     s3 = self.fully_connected3(a2)
    #     a3 = self.activation(s3)
    #     s4 = self.fully_connected4(a3)
    #     a4 = self.activation(s4)
    #     s5 = self.fully_connected5(a4)
    #     a5 = self.activation(s5)
    #     s6 = self.fully_connected6(a5)
    #     #s6 = nn.Softmax(s6)
    #     # WITHOUT ACTIVATION
    #     return s6, [s1, s2, s3, s4, s5, s6], [a0, a1, a2, a3, a4, a5]

    def forward_s_and_a(self, a0):
        # encoding:
        s1 = self.fully_connected1(a0)
        a1 = self.activation(s1)
        s2 = self.fully_connected2(a1)
        a2 = self.activation(s2)
        s3 = self.fully_connected3(a2)
        a3 = self.activation(s3)
        s4 = self.fully_connected4(a3)
        a4 = self.activation(s4)
        s5 = self.fully_connected5(a4)
        a5 = self.activation(s5)
        s6 = self.fully_connected6(a5)
        # I should have made 2 different networks for classification and regression at this point instead of
        # keep using the same with different parameters and conditions
        if self.s[-1] > 1: #only do softmax if we are doing classification. i.e. when the output dim is greater than 1
            return torch.softmax(s6, dim=1), [s1, s2, s3, s4, s5, s6], [a0, a1, a2, a3, a4, a5]
        return s6, [s1, s2, s3, s4, s5, s6], [a0, a1, a2, a3, a4, a5]

    def forward(self, x):
        x = self.activation(self.fully_connected1(x))
        x = self.activation(self.fully_connected2(x))
        x = self.activation(self.fully_connected3(x))
        x = self.activation(self.fully_connected4(x))
        x = self.activation(self.fully_connected5(x))
        x = self.fully_connected6(x)
        if self.s[-1] > 1: #only do softmax if we are doing classification. i.e. when the output dim is greater than 1
            return torch.softmax(x, dim=1)
        return x
        #return self.activation(self.decode8(x7))

    def forward_and_get_label(self, x):
        return torch.argmax(self.forward(x), dim=1)


    def initialize_weights_xavier(self, bias = 0.5):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, bias)


class drop_out_digits_classifier_kfac(nn.Module):
    def __init__(self, structure = [1000, 750, 500, 250 ,100, 10], input_size=784, activation_func = nn.Tanh(), bias = False):
        super(drop_out_digits_classifier_kfac, self).__init__()
        self.activation = activation_func
        self.s = structure

        # Inputs to hidden layer linear transformation
        self.dropout = nn.Dropout(0)
        self.fully_connected1 = nn.Linear(input_size, self.s[0], bias=bias)
        self.fully_connected2 = nn.Linear(self.s[0],  self.s[1], bias=bias)
        self.fully_connected3 = nn.Linear(self.s[1],  self.s[2], bias=bias)
        self.fully_connected4 = nn.Linear(self.s[2],  self.s[3], bias=bias)
        self.fully_connected5 = nn.Linear(self.s[3],  self.s[4], bias=bias)
        self.fully_connected6 = nn.Linear(self.s[4],  self.s[5], bias=bias)


    def forward_s_and_a(self, a0):
        # encoding:
        s1 = self.fully_connected1(a0)
        a1 = self.activation(s1)
        d = self.dropout(a1)

        s2 = self.fully_connected2(d)
        a2 = self.activation(s2)
        s3 = self.fully_connected3(a2)
        a3 = self.activation(s3)
        d = self.dropout(a3)

        s4 = self.fully_connected4(d)
        a4 = self.activation(s4)
        s5 = self.fully_connected5(a4)
        a5 = self.activation(s5)
        d = self.dropout(a5)

        s6 = self.fully_connected6(d)
        #s6 = nn.Softmax(s6)
        # WITHOUT ACTIVATION
        return s6, [s1, s2, s3, s4, s5, s6], [a0, a1, a2, a3, a4, a5]

    def forward(self, x):
        x = self.activation(self.fully_connected1(x))
        x = self.activation(self.fully_connected2(x))
        x = self.activation(self.fully_connected3(x))
        x = self.activation(self.fully_connected4(x))
        x = self.activation(self.fully_connected5(x))
        x = self.fully_connected6(x)
        #x = nn.Softmax(x)
        return x
        #return self.activation(self.decode8(x7))

    def forward_and_get_label(self, x):
        return torch.argmax(self.forward(x), dim=1)


    def initialize_weights_xavier(self, bias = 0.5):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, bias)



if __name__ == '__main__':
    # net = digits_autoencoder_kfac().cuda()
    # data = torch.rand(60000,784, dtype=torch.float32)
    # data2 = data[:30000].cuda()
    # for _ in range(10000):
    #     out, s, a= net.forward_s_and_a(data2)

    from helper_functions_v2 import inv
    import time
    d = torch.rand(1000, 1000, dtype=torch.float32, device="cuda:0")
    t1 = time.time()
    for _ in range(100):
        inv(d)
    torch.cuda.synchronize()
    print(time.time() - t1)

