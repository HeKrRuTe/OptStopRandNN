""" Computes the American option price by Randomized Least Square Monte Carlo.

It is the implementation of the Randomized Least Square Monte Carlo (RLSM)
introduced in (Optimal stopping via randomized neural networks,
Herrera, Krach, Ruyssen and Teichmann, 2021).
"""

import torch
from optimal_stopping.algorithms.backward_induction import regression
from optimal_stopping.algorithms.backward_induction import LSM


class ReservoirLeastSquarePricerFast(LSM.LeastSquaresPricer):
  """ Computes the American option price by Randomized Least Square Monte Carlo.
  """
  def __init__(self, model, payoff, hidden_size=10, factors=(1.,),
               nb_epochs=None, nb_batches=None, train_ITM_only=True):
      super().__init__(model, payoff, train_ITM_only=train_ITM_only)
      state_size = model.nb_stocks
      self.regression = regression.ReservoirLeastSquares(
          state_size, hidden_size, activation=torch.nn.LeakyReLU(factors[0]/2),
          factors=factors)


class ReservoirLeastSquarePricerFastRidge(LSM.LeastSquaresPricer):
  """ Computes the American option price by RLSM using ridge regression.
  """
  def __init__(self, model, payoff, hidden_size=10, factors=(1.,),
               ridge_coeff=1.,
               nb_epochs=None, nb_batches=None, train_ITM_only=True):
      super().__init__(model, payoff, train_ITM_only=train_ITM_only)
      state_size = model.nb_stocks
      self.regression = regression.ReservoirLeastSquaresRidge(
          state_size, hidden_size, activation=torch.nn.LeakyReLU(factors[0]/2),
          factors=factors, ridge_coeff=ridge_coeff)
