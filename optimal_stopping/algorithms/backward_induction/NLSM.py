""" Computes the American option price by Neural Least Square Monte Carlo.

It is the implementation of the Neural Least Square Monte Carlo (NLSM)
introduced in (Neural network regression for Bermudan option pricing,
Lapeyre and Lelong, 2019).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata

from optimal_stopping.algorithms.backward_induction import \
  backward_induction_pricer
from optimal_stopping.algorithms.utils import neural_networks


def init_weights(m):
  if isinstance(m, torch.nn.Linear):
    # torch.manual_seed(42)
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)



class NeuralNetworkPricer(backward_induction_pricer.AmericanOptionPricer):
  """ Computes the American option price by Neural Least Square Monte Carlo.

  It uses a neural network approximation to compute the continuation value.
  The weights of the neural network are optimized using gradient descent.
  """
  def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
               hidden_size=10, train_ITM_only=True, use_payoff_as_input=False):
    del nb_batches
    super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                     use_payoff_as_input=use_payoff_as_input)
    #neural regression class: train/evaluation of the neural network.
    self.neural_regression = NeuralRegression(
      model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1,
      model.nb_paths, hidden_size=hidden_size,
      nb_iters=nb_epochs)

  def calculate_continuation_value(
          self, values, immediate_exercise_value, stock_paths_at_timestep):
    """See base class."""
    inputs = stock_paths_at_timestep

    if self.train_ITM_only:
      in_the_money = np.where(immediate_exercise_value[:self.split] > 0)
      in_the_money_all = np.where(immediate_exercise_value > 0)
    else:
      in_the_money = np.where(immediate_exercise_value[:self.split] < np.infty)
      in_the_money_all = np.where(immediate_exercise_value < np.infty)
    continuation_values = np.zeros(stock_paths_at_timestep.shape[0])
    self.neural_regression.train_network(
      inputs[in_the_money[0]],
      values[in_the_money[0]])

    continuation_values[in_the_money_all[0]] = self.neural_regression.evaluate_network(
      inputs[in_the_money_all[0]])
    return continuation_values



class NeuralRegression(object):
  """ Train/evaluation of the neural network used for the continuation value.
  """
  def __init__(self, nb_stocks, nb_paths, hidden_size=10, nb_iters=20,
               batch_size=2000):
    self.batch_size = batch_size
    self.nb_stocks = nb_stocks
    self.nb_paths = nb_paths
    self.nb_iters = nb_iters
    self.neural_network = neural_networks.NetworkNLSM(
      self.nb_stocks, hidden_size=hidden_size).double()
    self.neural_network.apply(init_weights)

  def train_network(self, X_inputs, Y_labels):
    optimizer = optim.Adam(self.neural_network.parameters())
    X_inputs = torch.from_numpy(X_inputs).double()
    Y_labels = torch.from_numpy(Y_labels).double().view(len(Y_labels), 1)

    self.neural_network.train(True)
    for iteration in range(self.nb_iters):
      for batch in tdata.BatchSampler(
              tdata.RandomSampler(range(len(X_inputs)), replacement=False),
              batch_size=self.batch_size, drop_last=False):
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
          outputs = self.neural_network(X_inputs[batch])
          loss = nn.MSELoss(reduction="mean")(outputs, Y_labels[batch])
          loss.backward()
          optimizer.step()

  def evaluate_network(self, X_inputs):
    self.neural_network.train(False)
    X_inputs = torch.from_numpy(X_inputs).double()
    outputs = self.neural_network(X_inputs)
    return outputs.view(len(X_inputs)).detach().numpy()
