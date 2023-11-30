""" Computes the American option price using randomized fitted Q-Iteration.

It is the implementation of the randomized fitted Q-Iteration (RFQI)
introduced in (Optimal stopping via randomized neural networks,
Herrera, Krach, Ruyssen and Teichmann, 2021).
"""

import numpy as np
import torch

from optimal_stopping.algorithms.reinforcement_learning import \
  reinforcement_learning_price
from optimal_stopping.algorithms.reinforcement_learning import FQI
from optimal_stopping.algorithms.utils import randomized_neural_networks
from optimal_stopping.algorithms.utils import neural_networks
import sklearn.linear_model

class FQI_Reservoir(reinforcement_learning_price.FQI_RL):
  """Computes the American option price using randomized fitted Q-Iteration (RFQI)

  See fast version.
  """

  def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
               train_ITM_only=True,
               hidden_size=20, use_payoff_as_input=False):
    super().__init__(model, payoff, nb_epochs,
                     train_ITM_only=train_ITM_only,
                     use_payoff_as_input=use_payoff_as_input)
    del nb_batches
    self.model = model
    self.payoff = payoff
    self.dim_out = hidden_size
    self.nb_base_fcts = self.dim_out + 1
    self.state_size = self.model.nb_stocks*(1+self.use_var) + 2 + \
                      self.use_payoff_as_input*1
    # +2: additional for the time and time to maturity.
    self.reservoir = randomized_neural_networks.Reservoir(
      self.dim_out, self.state_size)
    self.reservoir2 = randomized_neural_networks.Reservoir2(
      self.dim_out, self.state_size)

  def evaluate_bases(self, stock_paths, path, date, nb_dates):
    stock_price_path_date = stock_paths[path, :, date]
    time = date / self.model.nb_dates
    time_to_maturity = 1 - time
    state = np.append(stock_price_path_date, [time, time_to_maturity])
    return self.reservoir.evaluate(state)


class FQI_ReservoirFast(FQI.FQIFast):
  """Computes the American option price using randomized fitted Q-Iteration (RFQI)"""

  def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
               hidden_size=20, factors=(1.,), train_ITM_only=True,
               use_payoff_as_input=False):
    super().__init__(model, payoff, nb_epochs,
                     train_ITM_only=train_ITM_only,
                     use_payoff_as_input=use_payoff_as_input)
    del nb_batches, train_ITM_only
    self.model = model
    self.payoff = payoff
    if hidden_size < 0:
      hidden_size = max(self.model.nb_stocks*abs(hidden_size), 5)
    else:
      self.dim_out = max(min(hidden_size, self.model.nb_stocks), 1)
    self.dim_out = hidden_size
    self.nb_base_fcts = self.dim_out + 1
    self.state_size = self.model.nb_stocks*(1+self.use_var) + 2 + \
                      self.use_payoff_as_input*1
    # +2: additional for the time and time to maturity.
    self.reservoir2 = randomized_neural_networks.Reservoir2(
      self.dim_out, self.state_size, activation=torch.nn.LeakyReLU(factors[0]/2))

  def evaluate_bases_all(self, stock_price):
    """ see base class"""
    stocks = torch.from_numpy(stock_price).type(torch.float32)
    stocks = stocks.permute(0, 2, 1)
    time = torch.linspace(0, 1, stocks.shape[1]).unsqueeze(0).repeat(
      (stocks.shape[0], 1)).unsqueeze(2)
    stocks = torch.cat([stocks, time, 1-time], dim=-1)
    random_base = self.reservoir2(stocks)
    random_base = torch.cat([random_base,
                             torch.ones([stocks.shape[0], stocks.shape[1], 1])],
                            dim=-1)
    return random_base.detach().numpy()


class FQI_ReservoirFastRidge(FQI.FQIFastRidge):
  """
  Computes the American option price using randomized fitted Q-Iteration (RFQI)
  with Ridge regression
  """

  def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
               hidden_size=20, factors=(1.,), train_ITM_only=True,
               ridge_coeff=1.,
               use_payoff_as_input=False):
    super().__init__(model, payoff, nb_epochs,
                     ridge_coeff=ridge_coeff,
                     train_ITM_only=train_ITM_only,
                     use_payoff_as_input=use_payoff_as_input)
    del nb_batches, train_ITM_only
    self.model = model
    self.payoff = payoff
    self.dim_out = max(min(hidden_size, self.model.nb_stocks), 1)
    # self.dim_out = hidden_size
    self.nb_base_fcts = self.dim_out + 1
    self.state_size = self.model.nb_stocks*(1+self.use_var) + 2 + \
                      self.use_payoff_as_input*1
    # +2: additional for the time and time to maturity.
    self.reservoir2 = randomized_neural_networks.Reservoir2(
      self.dim_out, self.state_size, activation=torch.nn.LeakyReLU(factors[0]/2))

  def evaluate_bases_all(self, stock_price):
    """ see base class"""
    stocks = torch.from_numpy(stock_price).type(torch.float32)
    stocks = stocks.permute(0, 2, 1)
    time = torch.linspace(0, 1, stocks.shape[1]).unsqueeze(0).repeat(
      (stocks.shape[0], 1)).unsqueeze(2)
    stocks = torch.cat([stocks, time, 1-time], dim=-1)
    random_base = self.reservoir2(stocks)
    random_base = torch.cat([random_base,
                             torch.ones([stocks.shape[0], stocks.shape[1], 1])],
                            dim=-1)
    return random_base.detach().numpy()


class FQI_ReservoirFastLasso(FQI_ReservoirFastRidge):
  """
  Computes the American option price using randomized fitted Q-Iteration (RFQI)
  with Lasso regression
  """
  def init_reg_model(self, ridge_coeff):
    self.reg_model = sklearn.linear_model.Lasso(
      alpha=ridge_coeff, fit_intercept=False)
    print("lasso")


class FQI_ReservoirFastTanh(FQI_ReservoirFast):
  """FQI_Reservoir"""

  def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
               hidden_size=20, factors=(1.,), train_ITM_only=True,
               use_payoff_as_input=False):
    super().__init__(model, payoff, nb_epochs, nb_batches, hidden_size,
                     train_ITM_only=train_ITM_only,
                     use_payoff_as_input=use_payoff_as_input)
    del nb_batches, train_ITM_only
    self.reservoir2 = randomized_neural_networks.Reservoir2(
      self.dim_out, self.state_size, factors=factors,
      activation=torch.nn.Tanh())


class FQI_ReservoirFastSoftplus(FQI_ReservoirFast):
  """FQI_Reservoir"""

  def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
               hidden_size=20, factors=(1.,), train_ITM_only=True,
               use_payoff_as_input=False):
    super().__init__(model, payoff, nb_epochs, nb_batches, hidden_size,
                     train_ITM_only=train_ITM_only,
                     use_payoff_as_input=use_payoff_as_input)
    del nb_batches, train_ITM_only
    self.reservoir2 = randomized_neural_networks.Reservoir2(
      self.dim_out, self.state_size, factors=factors,
      activation=torch.nn.Softplus())



class FQI_ReservoirFastRNN(FQI.FQIFast):
  """Computes the American option price using randomized recurrent fitted Q-Iteration (RRFQI)"""

  def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
               hidden_size=20, factors=(1., 1.), train_ITM_only=True,
               use_payoff_as_input=False):
    super().__init__(model, payoff, nb_epochs,
                     train_ITM_only=train_ITM_only,
                     use_payoff_as_input=use_payoff_as_input)
    del nb_batches, train_ITM_only
    self.model = model
    self.payoff = payoff
    self.dim_out = hidden_size
    self.nb_base_fcts = self.dim_out + 1
    self.state_size = self.model.nb_stocks*(1+self.use_var) + 2 + \
                      self.use_payoff_as_input*1
    # +2: additional for the time and time to maturity.
    self.reservoir2 = randomized_neural_networks.randomRNN(
      hidden_size=self.dim_out, state_size=self.state_size, factors=factors,
      extend=False)

  def evaluate_bases_all(self, stock_price):
    """ see base class"""
    stocks = torch.from_numpy(stock_price).type(torch.float32)
    stocks = stocks.permute(0, 2, 1)  # shape: [paths, dates, stocks]
    time = torch.linspace(0, 1, stocks.shape[1]).unsqueeze(0).repeat(
      (stocks.shape[0], 1)).unsqueeze(2)
    stocks = torch.cat([stocks, time, 1-time], dim=-1)
    stocks = stocks.permute(1, 0, 2)  # shape: [dates, paths, stocks]
    random_base = self.reservoir2(stocks)  # shape: [dates, paths, hidden]
    random_base = random_base.permute(1, 0, 2)  # shape: [paths, dates, hidden]
    random_base = torch.cat(
      [random_base, torch.ones(
        [random_base.shape[0], random_base.shape[1], 1])], dim=-1)
    return random_base.detach().numpy()
