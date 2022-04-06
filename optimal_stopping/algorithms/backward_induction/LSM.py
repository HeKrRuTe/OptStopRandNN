""" Computes the American option price by Least Square Monte Carlo (LSM).

It is the implementation of the Least Square Monte Carlo introduced in
(Valuing American Options by Simulation: A Simple Least-Squares Approach,
Longstaff and Schwartz, 2001).
"""

import numpy as np
from optimal_stopping.algorithms.backward_induction import regression
from optimal_stopping.algorithms.backward_induction import \
  backward_induction_pricer


class LeastSquaresPricer(backward_induction_pricer.AmericanOptionPricer):
  """ Computes the American option price by Least Square Monte Carlo (LSM).

  It uses a least square regression to compute the continuation value.
  The basis functions used are polynomial of order 2.
  """

  def __init__(self, model, payoff, nb_epochs=None, nb_batches=None,
               train_ITM_only=True, use_payoff_as_input=False):

    #regression class:  defines the regression used for the contination value.
    super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                     use_payoff_as_input=use_payoff_as_input)
    self.regression = regression.LeastSquares(
      model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1)

  def calculate_continuation_value(
          self, values, immediate_exercise_value, stock_paths_at_timestep):
    """See base class."""
    if self.train_ITM_only:
      in_the_money = np.where(immediate_exercise_value[:self.split] > 0)
      in_the_money_all = np.where(immediate_exercise_value > 0)
    else:
      in_the_money = np.where(immediate_exercise_value[:self.split] < np.infty)
      in_the_money_all = np.where(immediate_exercise_value < np.infty)
    return_values = np.zeros(stock_paths_at_timestep.shape[0])
    return_values[in_the_money_all[0]] = self.regression.calculate_regression(
      stock_paths_at_timestep, values,
      in_the_money, in_the_money_all
    )
    return return_values


class LeastSquarePricerDeg1(LeastSquaresPricer):
  """ Only polynomials of degree one are used for the least square regression."""
  def __init__(self, model, payoff, nb_epochs=None, nb_batches=None,
               train_ITM_only=True, use_payoff_as_input=False):
    super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                     use_payoff_as_input=use_payoff_as_input)
    self.regression = regression.LeastSquaresDeg1(
      model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1)


class LeastSquarePricerLaguerre(LeastSquaresPricer):
  """Least Square Monte Carlo using weighted Laguerre basis functions."""
  def __init__(self, model, payoff, nb_epochs=None, nb_batches=None,
               train_ITM_only=True, use_payoff_as_input=False):
    super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                     use_payoff_as_input=use_payoff_as_input)
    self.regression = regression.LeastSquaresLaguerre(
      model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1)


class LeastSquarePricerRidge(LeastSquaresPricer):
  """Least Square Monte Carlo using ridge regression."""
  def __init__(self, model, payoff, nb_epochs=None, nb_batches=None,
               train_ITM_only=True, ridge_coeff=1., use_payoff_as_input=False):
    super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                     use_payoff_as_input=use_payoff_as_input)
    self.regression = regression.LeastSquaresRidge(
      model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1,
      ridge_coeff=ridge_coeff)
