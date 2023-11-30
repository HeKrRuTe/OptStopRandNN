""" Calculate continuation values by regression.

It contains the different forms of regressions:
- Least squares regression using basis functions (for LSM)
- Least squares regression using randomized neural networks (for RLSM and RRLSM)

- Ridge regression using basis function or randomized neural networks.
- Least square using weighted Laguerre Polynomials.
"""

import numpy as np
from optimal_stopping.algorithms.utils import randomized_neural_networks
from optimal_stopping.algorithms.utils import basis_functions
import torch
import sklearn.linear_model

class Regression:
  def __init__(self, payoff_fct=None):
    pass

class LeastSquares(Regression):
  def __init__(self, nb_stocks):
    self.nb_stocks = nb_stocks
    self.bf = basis_functions.BasisFunctions(self.nb_stocks)


  def calculate_regression(self, X, Y, in_the_money, in_the_money_all):
    """ Calculate continuation values by least squares regression."""
    nb_paths, nb_stocks = X.shape
    reg_vect_mat = np.empty((nb_paths, self.bf.nb_base_fcts))
    for coeff in range(self.bf.nb_base_fcts):
      reg_vect_mat[:, coeff] = self.bf.base_fct(coeff, X[:, :], d2=True)
    coefficients = np.linalg.lstsq(
      reg_vect_mat[in_the_money[0]], Y[in_the_money[0]], rcond=None)
    continuation_values = np.dot(reg_vect_mat[in_the_money_all[0]],
                                 coefficients[0])
    return continuation_values


class LeastSquaresDeg1(LeastSquares):
  def __init__(self, nb_stocks):
    self.nb_stocks = nb_stocks
    self.bf = basis_functions.BasisFunctionsDeg1(self.nb_stocks)


class LeastSquaresLaguerre(LeastSquares):
  """ Calculate continuation values by least squares regression using
  weighted Laguerre polynomials.
  """
  def __init__(self, nb_stocks):
    self.nb_stocks = nb_stocks
    self.bf = basis_functions.BasisFunctionsLaguerre(self.nb_stocks)


class LeastSquaresRidge(Regression):
  """ Calculate continuation values by Ridge regression."""
  def __init__(self, nb_stocks, ridge_coeff=1.,):
    self.nb_stocks = nb_stocks
    self.bf = basis_functions.BasisFunctions(self.nb_stocks)
    self.alpha = ridge_coeff


  def calculate_regression(self, X, Y, in_the_money, in_the_money_all):
    nb_paths, nb_stocks = X.shape
    reg_vect_mat = np.empty((nb_paths, self.bf.nb_base_fcts))
    for coeff in range(self.bf.nb_base_fcts):
      reg_vect_mat[:, coeff] = self.bf.base_fct(coeff, X[:, :], d2=True)
    model = sklearn.linear_model.Ridge(alpha=self.alpha)
    model.fit(X=reg_vect_mat[in_the_money[0]], y=Y[in_the_money[0]])
    continuation_values = model.predict(reg_vect_mat[in_the_money_all[0]])
    return continuation_values


class ReservoirLeastSquares(Regression):
  def __init__(self, state_size, hidden_size=10):
    self.nb_base_fcts = hidden_size + 1
    self.state_size = state_size
    self.reservoir = randomized_neural_networks.Reservoir(
      hidden_size, self.state_size)

  def calculate_regression(self, X_unsorted, Y, in_the_money, in_the_money_all):
    X = X_unsorted
    nb_paths, nb_stocks = X.shape
    reg_vect_mat = np.empty((nb_paths, self.nb_base_fcts))
    for ipath in range(nb_paths):
      state = X[ipath, :]
      evaluated_nn = self.reservoir.evaluate(state)
      reg_vect_mat[ipath] = evaluated_nn
    coefficients = np.linalg.lstsq(
      reg_vect_mat[in_the_money[0]], Y[in_the_money[0]], rcond=None)
    continuation_values = np.dot(reg_vect_mat[in_the_money_all[0]],
                                 coefficients[0])
    return continuation_values



class ReservoirLeastSquares2(Regression):
  def __init__(self, state_size, hidden_size=10, factors=(1.,),
               activation=torch.nn.LeakyReLU(0.5), reinit=False):
    self.nb_base_fcts = hidden_size + 1
    self.state_size = state_size
    self.reinit = reinit
    self.reservoir = randomized_neural_networks.Reservoir2(
      hidden_size, self.state_size, factors=factors, activation=activation)

  def calculate_regression(self, X_unsorted, Y, in_the_money, in_the_money_all,
                           coefficients=None, return_coefficients=False):
    if self.reinit:
      self.reservoir.init()
    X = torch.from_numpy(X_unsorted)
    X = X.type(torch.float32)
    reg_input = np.concatenate(
      [self.reservoir(X).detach().numpy(), np.ones((len(X), 1))], axis=1)
    if coefficients is None:
      coefficients = np.linalg.lstsq(
        reg_input[in_the_money[0]], Y[in_the_money[0]], rcond=None)
    continuation_values = np.dot(reg_input[in_the_money_all[0]], coefficients[0])
    if return_coefficients:
      return continuation_values, coefficients
    return continuation_values


class ReservoirLeastSquaresRidge(Regression):
  """ Calculate continuation values by Ridge regression using randomized NN.
  """
  def __init__(self, state_size, hidden_size=10, factors=(1.,), ridge_coeff=1.,
               activation=torch.nn.LeakyReLU(0.5)):
    self.nb_base_fcts = hidden_size + 1
    self.state_size = state_size
    self.alpha = ridge_coeff
    self.reservoir = randomized_neural_networks.Reservoir2(
      hidden_size, self.state_size, factors=factors, activation=activation)

  def calculate_regression(self, X_unsorted, Y, in_the_money, in_the_money_all):
    X = torch.from_numpy(X_unsorted)
    X = X.type(torch.float32)
    reg_input = np.concatenate(
      [self.reservoir(X).detach().numpy(), np.ones((len(X), 1))], axis=1)

    model = sklearn.linear_model.Ridge(alpha=self.alpha)
    model.fit(X=reg_input[in_the_money[0]], y=Y[in_the_money[0]])
    continuation_values = model.predict(reg_input[in_the_money_all[0]])
    return continuation_values
