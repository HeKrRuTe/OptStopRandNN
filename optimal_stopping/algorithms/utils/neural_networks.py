import torch.nn as nn


class NetworkNLSM(nn.Module):
  def __init__(self, nb_stocks, hidden_size=10):
    super(NetworkNLSM, self).__init__()
    H = hidden_size
    self.layer1 = nn.Linear(nb_stocks, H)
    self.leakyReLU = nn.LeakyReLU(0.5)
    self.sigmoid = nn.Sigmoid()
    self.bn1 = nn.BatchNorm1d(num_features=H)
    self.layer2 = nn.Linear(H, H)
    self.bn2 = nn.BatchNorm1d(num_features=H)
    self.layer3 = nn.Linear(H, 1)
    self.bn3 = nn.BatchNorm1d(num_features=1)
    self.Softplus = nn.Softplus()
    self.tanh = nn.Tanh()

  def forward(self, x):
    x = self.layer1(x)
    x = self.leakyReLU(x)
    x = self.layer3(x)
    return x


class NetworkDOS(nn.Module):
  def __init__(self, nb_stocks, hidden_size=10):
    super(NetworkDOS, self).__init__()
    H = hidden_size
    self.bn0 = nn.BatchNorm1d(num_features=nb_stocks)
    self.layer1 = nn.Linear(nb_stocks, H)
    self.leakyReLU = nn.LeakyReLU(0.5)
    self.Softplus = nn.Softplus()
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
    self.bn1 = nn.BatchNorm1d(num_features=H)
    self.layer2 = nn.Linear(H, H)
    self.bn2 = nn.BatchNorm1d(num_features=H)
    self.layer3 = nn.Linear(H, 1)
    self.bn3 = nn.BatchNorm1d(num_features=1)

  def forward(self, x):
    x = self.bn0(x)
    x = self.layer1(x)
    x = self.relu(x)
    x = self.bn2(x)
    x = self.layer3(x)
    x = self.sigmoid(x)
    return x




