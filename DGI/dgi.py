import torch
import torch.nn as nn
from layers import GCN, Readout, Discriminator

class DGI(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(DGI, self).__init__()
        self.gcn = GCN(n_features, n_hidden)
        self.readout = Readout()
        self.sigmoid = nn.Sigmoid()
        self.discriminator = Discriminator(n_hidden)

    def forward(self, x_1, x_2, adj):
        h_1 = self.gcn(x_1, adj)
        h_2 = self.gcn(x_2, adj)

        s = self.readout(h_1)
        s = self.sigmoid(s)
        score = self.discriminator(s, h_1, h_2)

        return score

    def embed(self, h, adj):
        h = self.gcn(h, adj)
        s = self.readout(h)

        return h.detach()

    def patch_representation(self, feature, adj):
        h_i = self.gcn(feature, adj)
        return h_i


