import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_shape, n_endmembers):
        super(Autoencoder, self).__init__()

        # encoder
        self.encoder = NotImplemented

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_endmembers, input_shape, bias=False)
        )

    def encode(self, x):

        out = self.encoder(x)

        # Enforce that abundances sum to 1 (ASC)
        out = out / out.sum(dim=1, keepdim=True)

        return out

    def forward(self, x):

        out = self.encode(x)
        out = self.decoder(out)

        return out


class Net1(Autoencoder):
    def __init__(self, input_shape, n_endmembers):
        super(Net1, self).__init__(input_shape, n_endmembers)

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, n_endmembers),
            nn.Sigmoid()
        )


class Net2(Autoencoder):
    def __init__(self, input_shape, n_endmembers, Dropoutp):
        super(Net2, self).__init__(input_shape, n_endmembers)

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, n_endmembers),
            nn.ReLU(),
            nn.Dropout(Dropoutp)
        )


class  Net3(Autoencoder):
    def __init__(self, input_shape, n_endmembers):
        super(Net3, self).__init__(input_shape, n_endmembers)

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, n_endmembers),
            nn.ReLU(),
            nn.Linear(n_endmembers, n_endmembers),
            nn.ReLU()
        )
