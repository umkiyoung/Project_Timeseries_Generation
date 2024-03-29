import torch
import torch.nn as nn
import numpy as np
from scipy.stats import multivariate_normal
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Embedder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        r_out, _ = self.rnn(x)
        output = torch.sigmoid(self.fc(r_out))
        return output


class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        r_out, _ = self.rnn(h)
        output = torch.sigmoid(self.fc(r_out))
        return output

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z):
        r_out, _ = self.rnn(z)
        output = torch.sigmoid(self.fc(r_out))
        return output


class Supervisor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers-1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):
        r_out, _ = self.rnn(h)
        output = torch.sigmoid(self.fc(r_out))
        return output
    
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, h):
        r_out, _ = self.rnn(h)
        output = torch.sigmoid(self.fc(r_out))
        return output

