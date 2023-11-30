""" Computes the American option price by Randomized Least Square Monte Carlo.

It is the implementation of the Randomized Least Square Monte Carlo (RLSM)
introduced in (Optimal stopping via randomized neural networks,
Herrera, Krach, Ruyssen and Teichmann, 2021).
"""

import torch
from optimal_stopping.algorithms.backward_induction import regression
from optimal_stopping.algorithms.backward_induction import LSM



class ReservoirLeastSquarePricer(LSM.LeastSquaresPricer):
  """ Computes the American option price by Randomized Least Square Monte Carlo.

  It is the first version. It was optimzed later in order to be fasterselfself.
  See "ReservoirLeastSquarePricerFast".
  """
  def __init__(self, model, payoff, hidden_size=10, nb_epochs=None,
               nb_batches=None, train_ITM_only=True, use_payoff_as_input=False):
      super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                       use_payoff_as_input=use_payoff_as_input)
      if hidden_size < 0:
          hidden_size = 50 + abs(hidden_size)*model.nb_stocks
      state_size = model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1
      self.regression = regression.ReservoirLeastSquares(
          state_size, hidden_size)


class ReservoirLeastSquarePricerFast(LSM.LeastSquaresPricer):
  """ Computes the American option price by Randomized Least Square Monte Carlo.
  """
  def __init__(self, model, payoff, hidden_size=10, factors=(1.,),
               nb_epochs=None, nb_batches=None, train_ITM_only=True,
               use_payoff_as_input=False):
      super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                       use_payoff_as_input=use_payoff_as_input)
      if hidden_size < 0:
          hidden_size = 50 + abs(hidden_size)*model.nb_stocks
      state_size = model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1
      self.regression = regression.ReservoirLeastSquares2(
          state_size, hidden_size, activation=torch.nn.LeakyReLU(factors[0]/2),
          factors=factors[1:])


class ReservoirLeastSquarePricerFastTanh(LSM.LeastSquaresPricer):
  """ Computes the American option price by RLSM using activation function tanh.
  """
  def __init__(self, model, payoff, hidden_size=10, factors=(1.,),
               nb_epochs=None, nb_batches=None, train_ITM_only=True,
               use_payoff_as_input=False):
      super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                       use_payoff_as_input=use_payoff_as_input)
      if hidden_size < 0:
          hidden_size = 50 + abs(hidden_size)*model.nb_stocks
      state_size = model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1
      self.regression = regression.ReservoirLeastSquares2(
          state_size, hidden_size, factors=factors, activation=torch.nn.Tanh())


class ReservoirLeastSquarePricerFastSoftplus(LSM.LeastSquaresPricer):
    """ Computes the American option price by RLSM using activation function tanh.
    """
    def __init__(self, model, payoff, hidden_size=10, factors=(1.,),
                 nb_epochs=None, nb_batches=None, train_ITM_only=True,
                 use_payoff_as_input=False):
        super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                         use_payoff_as_input=use_payoff_as_input)
        if hidden_size < 0:
            hidden_size = 50 + abs(hidden_size) * model.nb_stocks
        state_size = model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1
        self.regression = regression.ReservoirLeastSquares2(
            state_size, hidden_size, factors=factors,
            activation=torch.nn.Softplus(beta=factors[1]))


class ReservoirLeastSquarePricerFastSoftplusReinit(LSM.LeastSquaresPricer):
    """ Computes the American option price by RLSM using activation function tanh.
    """
    def __init__(self, model, payoff, hidden_size=10, factors=(1.,),
                 nb_epochs=None, nb_batches=None, train_ITM_only=True,
                 use_payoff_as_input=False):
        super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                         use_payoff_as_input=use_payoff_as_input)
        if hidden_size < 0:
            hidden_size = 50 + abs(hidden_size) * model.nb_stocks
        state_size = model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1
        self.regression = regression.ReservoirLeastSquares2(
            state_size, hidden_size, factors=factors,
            activation=torch.nn.Softplus(beta=factors[1]), reinit=True)


class ReservoirLeastSquarePricerFastGELU(LSM.LeastSquaresPricer):
    """ Computes the American option price by RLSM using activation function tanh.
    """
    def __init__(self, model, payoff, hidden_size=10, factors=(1.,),
                 nb_epochs=None, nb_batches=None, train_ITM_only=True,
                 use_payoff_as_input=False):
        super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                         use_payoff_as_input=use_payoff_as_input)
        if hidden_size < 0:
            hidden_size = 50 + abs(hidden_size) * model.nb_stocks
        state_size = model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1
        self.regression = regression.ReservoirLeastSquares2(
            state_size, hidden_size, factors=factors,
            activation=torch.nn.GELU())


class ReservoirLeastSquarePricerFastSILU(LSM.LeastSquaresPricer):
    """ Computes the American option price by RLSM using activation function tanh.
    """
    def __init__(self, model, payoff, hidden_size=10, factors=(1.,),
                 nb_epochs=None, nb_batches=None, train_ITM_only=True,
                 use_payoff_as_input=False):
        super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                         use_payoff_as_input=use_payoff_as_input)
        if hidden_size < 0:
            hidden_size = 50 + abs(hidden_size) * model.nb_stocks
        state_size = model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1
        self.regression = regression.ReservoirLeastSquares2(
            state_size, hidden_size, factors=factors,
            activation=torch.nn.SiLU())


class ReservoirLeastSquarePricerFastELU(LSM.LeastSquaresPricer):
    """ Computes the American option price by RLSM using activation function tanh.
    """
    def __init__(self, model, payoff, hidden_size=10, factors=(1.,),
                 nb_epochs=None, nb_batches=None, train_ITM_only=True,
                 use_payoff_as_input=False):
        super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                         use_payoff_as_input=use_payoff_as_input)
        if hidden_size < 0:
            hidden_size = 50 + abs(hidden_size) * model.nb_stocks
        state_size = model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1
        self.regression = regression.ReservoirLeastSquares2(
            state_size, hidden_size, factors=factors,
            activation=torch.nn.ELU(alpha=factors[1]))


class ReservoirLeastSquarePricerFastRidge(LSM.LeastSquaresPricer):
  """ Computes the American option price by RLSM using ridge regression.
  """
  def __init__(self, model, payoff, hidden_size=10, factors=(1.,),
               ridge_coeff=1.,
               nb_epochs=None, nb_batches=None, train_ITM_only=True,
               use_payoff_as_input=False):
      super().__init__(model, payoff, train_ITM_only=train_ITM_only,
                       use_payoff_as_input=use_payoff_as_input)
      if hidden_size < 0:
          hidden_size = 50 + abs(hidden_size)*model.nb_stocks
      state_size = model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1
      self.regression = regression.ReservoirLeastSquaresRidge(
          state_size, hidden_size, activation=torch.nn.LeakyReLU(factors[0]/2),
          factors=factors, ridge_coeff=ridge_coeff)
