import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(GCN, self).__init__()
        self.fc = nn.Linear(n_input, n_hidden)
        self.activation = nn.PReLU()
        nn.init.xavier_normal_(self.fc.weight.data)

    def forward(self, x, adj):
        x = self.fc(x)
        x = torch.unsqueeze(torch.spmm(adj, torch.squeeze(x, 0)), 0)
        x = self.activation(x)

        return x

class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, h):
        return torch.mean(h, dim=1)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        nn.init.xavier_normal_(self.f_k.weight.data)

    def forward(self, s, h_1, h_2):
        expanded_s = torch.unsqueeze(s, 1).expand_as(h_1)
        score_1 = torch.squeeze(self.f_k(h_1, expanded_s), 2)
        score_2 = torch.squeeze(self.f_k(h_2, expanded_s), 2)
        logits = torch.cat([score_1, score_2], dim=1)

        return logits

class LogisticRegression(nn.Module):
    def __init__(self, n_hidden, n_class):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(n_hidden, n_class)
        nn.init.xavier_normal_(self.fc.weight.data)

    def forward(self, h):
        return torch.squeeze(self.fc(h))