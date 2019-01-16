import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_shape, n_endmembers):
        super(Autoencoder, self).__init__()

        # encoder
        self.encoder = NotImplemented

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_endmembers, input_shape),
            nn.ReLU()
        )

    def forward(self, x):

        out = self.encoder(x)
        out = self.decoder(out)

        return out

    def encode(self, x):
        return self.encoder(x)


class Basic(Autoencoder):
    def __init__(self, input_shape, n_endmembers):
        super(Basic, self).__init__(input_shape, n_endmembers)

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, n_endmembers),
            nn.ReLU()
        )


class Deep1(Autoencoder):
    def __init__(self, input_shape, n_endmembers):
        super(Deep1, self).__init__(input_shape, n_endmembers)

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, n_endmembers),
            nn.ReLU(),
            nn.Linear(n_endmembers, n_endmembers),
            nn.ReLU()
        )
