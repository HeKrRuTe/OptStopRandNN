"""Base class that computes American option prices using reinforcement learning.

All reinforcement learning based algorithms such as
FQI (fitted Q-Iterration),
RFQI (Randomized fitted Q-Iterration),
LSPI (Least squares) are inherited from this class.
"""

import math, time
import numpy as np

from optimal_stopping.algorithms.backward_induction import \
  backward_induction_pricer
from optimal_stopping.algorithms.utils import neural_networks
from optimal_stopping.run import configs


class FQI_RL(backward_induction_pricer.AmericanOptionPricer):
  """fitted Q-Iterration base class"""

  def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
               train_ITM_only=True, use_payoff_as_input=False):
    del nb_batches
    self.model = model
    self.use_var = False
    if self.model.return_var:
      self.use_var = True
    self.payoff = payoff
    self.nb_epochs = nb_epochs
    self.nb_base_fcts = 0
    self.use_payoff_as_input = use_payoff_as_input
    self.train_ITM_only = train_ITM_only


  def get_indicator_stop(self, payoff, continuation_value):
    return max(payoff, continuation_value)

  def get_contribution_u(
      self, payoff, evaluated_bases, next_evaluated_bases, discount_factor, continuation_value):
    del payoff
    del discount_factor
    del continuation_value
    return np.outer(evaluated_bases, evaluated_bases)

  def evaluate_bases(self, stock_paths, path, date, nb_dates):
    raise NotImplementedError

  def evaluate_bases_all(self, stock_paths):
    """
    Args:
     stock_price (np.array, shape [nb_paths, nb_stocks, nb_dates])

    Returns:
     evaluated basis functions
      (np.array, shape [nb_paths, nb_dates, nb_base_fcts])
    """
    raise NotImplementedError

  def price(self):
    t1 = time.time()
    if configs.path_gen_seed.get_seed() is not None:
      np.random.seed(configs.path_gen_seed.get_seed())
    stock_paths, var_paths = self.model.generate_paths()
    payoffs = self.payoff(stock_paths)
    stock_paths_with_payoff = np.concatenate(
      [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
    self.split = int(len(stock_paths)/2)
    time_for_path_gen = time.time() - t1
    print("time path gen: {}".format(time.time() - t1), end=" ")
    nb_paths, nb_stocks, nb_dates = stock_paths[:self.split].shape
    matrixU = np.zeros((self.nb_base_fcts, self.nb_base_fcts), dtype=float)
    vectorV = np.zeros(self.nb_base_fcts, dtype=float)
    weights = np.zeros(self.nb_base_fcts, dtype=float)
    deltaT = self.model.maturity / nb_dates
    discount_factor = math.exp(-self.model.rate * deltaT)
    if self.use_payoff_as_input:
      paths = stock_paths_with_payoff
    else:
      paths = stock_paths
    if self.use_var:
      paths = np.concatenate([paths, var_paths], axis=1)

    for epoch in range(self.nb_epochs):
      for i_path, path in enumerate(range(nb_paths)):
        for date in range(nb_dates - 1):
          payoff = self.payoff([stock_paths[path, :, date+1]])[0]
          evaluated_bases = self.evaluate_bases(paths, path, date, nb_dates)
          next_evaluated_bases  = self.evaluate_bases(paths, path,  date+1, nb_dates)
          continuation_value = np.inner(weights, next_evaluated_bases)
          indicator_stop = self.get_indicator_stop(payoff, continuation_value)
          contribution_u = self.get_contribution_u(
              payoff, evaluated_bases, next_evaluated_bases, discount_factor, continuation_value)
          matrixU += contribution_u
          vectorV += evaluated_bases * np.asarray(discount_factor) * np.asarray(indicator_stop)

      weights = np.linalg.solve(matrixU, vectorV)

    nb_paths, nb_stocks, nb_dates = stock_paths.shape
    prices = np.zeros(nb_paths, dtype=float)
    for path in range(nb_paths):
      for date in range(nb_dates):
        evaluated_bases = self.evaluate_bases(paths, path, date, nb_dates)
        payoff = self.payoff([stock_paths[path, :, date]])[0]
        continuation_value = np.inner(weights, evaluated_bases)
        continuation_value = max(continuation_value, 0)
        if payoff > continuation_value or (date == nb_dates-1):
          prices[path] = payoff * (discount_factor ** date)
          break
    return np.mean(prices[self.split:]), time_for_path_gen
