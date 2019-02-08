import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Network design from paper:

Layer #     Layer type      Activation func.        # units

1           Input           Linear                  B
2           Hidden          g                       9R
3           Hidden          g                       6R
4           Hidden          g                       3R
5           Hidden          g                       R
6           Batch Norm      -                       R
7           Thresholding    RelU / LRelU            R
8           Enforce ASC     -                       R
9           Guassian Dropout

"""


class Autoencoder(nn.Module):
    def __init__(self, input_shape, n_endmembers):
        super(Autoencoder, self).__init__()

        # encoder
        self.encoder = NotImplemented

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_endmembers, input_shape, bias=False),
        )

    def forward(self, x):

        x = self.encoder(x)

        # Enforce that abundances sum to 1 (ASC)
        encoded = x / x.sum(dim=1, keepdim=True)

        out = self.decoder(encoded)

        return encoded, out


class Net1(Autoencoder):
    def __init__(self, input_shape, n_endmembers):
        super(Net1, self).__init__(input_shape, n_endmembers)

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, n_endmembers),
            nn.Sigmoid(),
        )


class Net2(Autoencoder):
    def __init__(self, input_shape, n_l1, n_endmembers):
        super(Net2, self).__init__(input_shape, n_endmembers)

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, n_l1),
            nn.ReLU(),
            nn.Linear(n_l1, n_endmembers),
            nn.ReLU()
        )


class Net3(Autoencoder):
    def __init__(self, input_shape, n_l1, n_l2, n_endmembers):
        super(Net3, self).__init__(input_shape, n_endmembers)

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, n_l1),
            nn.ReLU(),
            nn.Linear(n_l1, n_l2),
            nn.ReLU(),
            nn.Linear(n_l2, n_endmembers),
            nn.ReLU()
        )

if __name__ == '__main__':

    model = Net1(300, 5)

    # model
    print(model)

    # params
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
