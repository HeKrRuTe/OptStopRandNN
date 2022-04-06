""" Computes the American option price using the deep optimal stopping (DOS).

It is the implementation of the deep optimal stopping (DOS) introduced in
(deep optimal stopping, Becker, Cheridito and Jentzen, 2020).
"""

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as tdata

from optimal_stopping.algorithms.backward_induction import \
  backward_induction_pricer
from optimal_stopping.algorithms.utils import neural_networks


def init_weights(m):
  if isinstance(m, torch.nn.Linear):
    torch.manual_seed(42)
    # torch.nn.init.zeros_(m.weight)
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)



class DeepOptimalStopping(backward_induction_pricer.AmericanOptionPricer):
  """Computes the American option price using the deep optimal stopping (DOS)
  """

  def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
               hidden_size=10, use_path=False, use_payoff_as_input=False):
    del nb_batches
    super().__init__(model, payoff, use_path=use_path,
                     use_payoff_as_input=use_payoff_as_input)
    if self.use_path:
      state_size = (model.nb_stocks*(1+self.use_var) +
                    self.use_payoff_as_input*1) * \
                   (model.nb_dates+1)
    else:
      state_size = model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1
    self.neural_stopping = OptimalStoppingOptimization(
      state_size, model.nb_paths, hidden_size=hidden_size,
      nb_iters=nb_epochs)

  def stop(self, stock_values, immediate_exercise_values,
           discounted_next_values, h=None, var_paths=None,
           train=True, return_continuation_values=False):
    """ see base class """
    if self.use_var:
      stock_values = np.concatenate([stock_values, var_paths], axis=1)
    if self.use_path:
      # shape [paths, stocks, dates up to now]
      stock_values = np.flip(stock_values, axis=2)
      # add zeros to get shape [paths, stocks, dates+1]
      stock_values = np.concatenate(
        [stock_values, np.zeros(
          (stock_values.shape[0], stock_values.shape[1],
           self.model.nb_dates + 1 - stock_values.shape[2]))], axis=-1)
      stock_values = stock_values.reshape((stock_values.shape[0], -1))
    if train:
      self.neural_stopping.train_network(
        stock_values[:self.split],
        immediate_exercise_values.reshape(-1, 1)[:self.split],
        discounted_next_values[:self.split])

    inputs = stock_values
    stopping_rule = self.neural_stopping.evaluate_network(inputs)
    if return_continuation_values:
      return stopping_rule, None
    return stopping_rule




class OptimalStoppingOptimization(object):
  """Train/evaluation of the neural network used for the stopping decision"""

  def __init__(self, nb_stocks, nb_paths, hidden_size=10, nb_iters=20,
               batch_size=2000):
    self.nb_stocks = nb_stocks
    self.nb_paths = nb_paths
    self.nb_iters = nb_iters
    self.batch_size = batch_size
    self.network = neural_networks.NetworkDOS(
      self.nb_stocks, hidden_size=hidden_size).double()
    self.network.apply(init_weights)

  def _Loss(self, X):
    return -torch.mean(X)

  def train_network(self, stock_values, immediate_exercise_value,
                    discounted_next_values):
    optimizer = optim.Adam(self.network.parameters())
    discounted_next_values = torch.from_numpy(discounted_next_values).double()
    immediate_exercise_value = torch.from_numpy(immediate_exercise_value).double()
    inputs = stock_values
    X_inputs = torch.from_numpy(inputs).double()

    self.network.train(True)
    ones = torch.ones(len(discounted_next_values))
    for iteration in range(self.nb_iters):
      for batch in tdata.BatchSampler(
              tdata.RandomSampler(range(len(X_inputs)), replacement=False),
              batch_size=self.batch_size, drop_last=False):

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
          outputs = self.network(X_inputs[batch]).reshape(-1)
          values = (immediate_exercise_value[batch].reshape(-1) * outputs +
                    discounted_next_values[batch] * (ones[batch] - outputs))
          loss = self._Loss(values)
          loss.backward()
          optimizer.step()

  def evaluate_network(self, X_inputs):
    self.network.train(False)
    X_inputs = torch.from_numpy(X_inputs).double()
    outputs = self.network(X_inputs)
    return outputs.view(len(X_inputs)).detach().numpy()
