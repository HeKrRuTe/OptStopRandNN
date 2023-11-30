"""Base class that computes the American option price using backward recusrion.

All algorithms that are using a backward recursion such as
LSM (Least squares Monte Carlo),
NLSM (Neural Least squares Monte Carlo),
RLSM (Randomized Least squares Monte Carlo)
and DOS (Deep Optimal Stopping) are inherited from this class.
"""

import numpy as np
import time
import optimal_stopping.data.stock_model as stock_model
import optimal_stopping.algorithms.utils.utilities as utilities
from optimal_stopping.run import configs
import copy

import scipy.optimize as opt

from sklearn.linear_model import LinearRegression


class AmericanOptionPricer:
  """Computes the price of an American Option using backward recusrion.
  """
  def __init__(self, model, payoff, use_rnn=False, train_ITM_only=True,
               use_path=False, use_payoff_as_input=False):

    #class model: The stochastic process model of the stock (e.g. Black Scholes).
    self.model = model
    self.use_var = False
    if self.model.return_var:
      self.use_var = True

    #class payoff: The payoff function of the option (e.g. Max call).
    self.payoff = payoff

    #bool: randomized neural network is replaced by a randomized recurrent NN.
    self.use_rnn = use_rnn

    #bool: x_k is replaced by the entire path (x_0, .., x_k) as input of the NN.
    self.use_path = use_path

    #bool: only the paths that are In The Money (ITM) are used for the training.
    self.train_ITM_only = train_ITM_only

    #bool: whether to use the payoff as extra input in addition to stocks
    self.use_payoff_as_input = use_payoff_as_input

    #int: used for ONLSM, tells model which weight to use
    self.which_weight = 0

  def calculate_continuation_value(
          self, values, immediate_exercise_value, stock_paths_at_timestep):
    """Computes the continuation value of an american option at a given date.

    All algorithms that inherited from this class (AmericanOptionPricer) where
    the continuation value is approximated by basis functions (LSM),
    neural networks (NLSM), randomized neural networks (RLSM), or
    recurrent randomized neural networks (RRLSM) only differ by a this function.

    The number of paths determines the size of the arrays.

    Args:
      values (np array): the option price of the next date (t+1).
      immediate_exercise_value (np array): the payoff evaluated with the current
       stock price (date t).
      stock_paths_at_timestep (np array): The stock price at the current date t.

    Returns:
      np array: the option price at current date t if we continue until next
       date t+1.
    """
    raise NotImplementedError

  def stop(self, stock_values, immediate_exercise_values,
           discounted_next_values, h=None, var_paths=None,
           return_continuation_values=False):
    """Returns a vector of {0, 1}s (one per path) for a given data, where:
        1 means stop, and
        0 means continue.

    The optimal stopping algorithm (DOS) where the optimal stopping is
    approximated by a neural network has a different function "stop".
    """
    stopping_rule = np.zeros(len(stock_values))
    if self.use_rnn:
      continuation_values = self.calculate_continuation_value(
          discounted_next_values,
          immediate_exercise_values, h)
    else:
      if self.use_var:
        stock_values = np.concatenate([stock_values, var_paths], axis=1)
      continuation_values = self.calculate_continuation_value(
        discounted_next_values,
        immediate_exercise_values, stock_values)
    if self.train_ITM_only:
      which = (immediate_exercise_values > continuation_values) & \
              (immediate_exercise_values > np.finfo(float).eps)
    else:
      which = immediate_exercise_values > continuation_values
    stopping_rule[which] = 1
    if return_continuation_values:
      return stopping_rule, continuation_values
    return stopping_rule

  def price(self, train_eval_split=2):
    """
    Compute the price of an American Option using a backward recursion.
    """
    model = self.model
    t1 = time.time()
    if configs.path_gen_seed.get_seed() is not None:
      np.random.seed(configs.path_gen_seed.get_seed())
    stock_paths, var_paths = self.model.generate_paths()
    payoffs = self.payoff(stock_paths)
    stock_paths_with_payoff = np.concatenate(
      [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
    time_for_path_gen = time.time() - t1
    self.split = int(len(stock_paths)/train_eval_split)
    print("time path gen: {}".format(time_for_path_gen), end=" ")
    if self.use_rnn:
      if self.use_payoff_as_input:
        hs = self.compute_hs(stock_paths_with_payoff, var_paths=var_paths)
      else:
        hs = self.compute_hs(stock_paths, var_paths=var_paths)
    disc_factor = np.math.exp((-model.rate) * model.maturity /
               (model.nb_dates))
    immediate_exercise_value = self.payoff.eval(stock_paths[:, :, -1])
    values = immediate_exercise_value
    for i, date in enumerate(range(stock_paths.shape[2] - 2, 0, -1)):
      self.which_weight = i
      immediate_exercise_value = self.payoff.eval(stock_paths[:, :, date])
      if self.use_rnn:
        h = hs[date]
      else:
        h = None
      if self.use_path:
        varp = None
        if self.use_var:
          varp = var_paths[:, :, :date+1]
        if self.use_payoff_as_input:
          paths = stock_paths_with_payoff[:, :, :date+1]
        else:
          paths = stock_paths[:, :, :date+1]
      else:
        varp = None
        if self.use_var:
          varp = var_paths[:, :, date]
        if self.use_payoff_as_input:
          paths = stock_paths_with_payoff[:, :, date]
        else:
          paths = stock_paths[:, :, date]
      stopping_rule = self.stop(
        paths, immediate_exercise_value,
        values*disc_factor, h=h, var_paths=varp)
      which = stopping_rule > 0.5
      values[which] = immediate_exercise_value[which]
      values[~which] *= disc_factor
    payoff_0 = self.payoff.eval(stock_paths[:, :, 0])[0]
    return max(payoff_0, np.mean(values[self.split:]) * disc_factor), \
           time_for_path_gen

  def price_upper_lower_bound(self, verbose=0, train_eval_split=2):
    """
    Compute upper and lower bounds of the price of an American Option using a
    backward recursion.
    """
    model = self.model
    if self.train_ITM_only:
        raise ValueError("train_ITM_only is not supported for upper/lower bound")
    t1 = time.time()
    if configs.path_gen_seed.get_seed() is not None:
      np.random.seed(configs.path_gen_seed.get_seed())
    stock_paths, var_paths = self.model.generate_paths()
    payoffs = self.payoff(stock_paths)
    power = np.arange(0, model.nb_dates+1)
    disc_factor = np.exp(
      (-model.rate) * model.maturity / model.nb_dates * power)
    disc_factors = np.repeat(
      np.expand_dims(disc_factor, axis=0), repeats=payoffs.shape[0], axis=0)
    payoffs = payoffs * disc_factors
    stock_paths_with_payoff = np.concatenate(
      [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
    time_for_path_gen = time.time() - t1
    self.split = int(len(stock_paths)/train_eval_split)
    M_diff = np.zeros((stock_paths.shape[0], stock_paths.shape[2]))
    print("time path gen: {}".format(time_for_path_gen), end=" ")
    if self.use_rnn:
      if self.use_payoff_as_input:
        hs = self.compute_hs(stock_paths_with_payoff, var_paths=var_paths)
      else:
        hs = self.compute_hs(stock_paths, var_paths=var_paths)

    immediate_exercise_value = payoffs[:, -1]
    values = copy.deepcopy(immediate_exercise_value)
    prev_cont_val = np.zeros_like(values)
    prev_values = copy.deepcopy(immediate_exercise_value)
    prevprev_values = copy.deepcopy(immediate_exercise_value)
    for i, date in enumerate(range(stock_paths.shape[2] - 2, -1, -1)):
      if verbose:
        print(date)
      self.which_weight = i
      immediate_exercise_value = payoffs[:, date]
      if self.use_path:
        varp = None
        if self.use_var:
          varp = var_paths[:, :, :date+1]
        if self.use_payoff_as_input:
          paths = stock_paths_with_payoff[:, :, :date+1]
        else:
          paths = stock_paths[:, :, :date+1]
      else:
        varp = None
        if self.use_var:
          varp = var_paths[:, :, date]
        if self.use_payoff_as_input:
          paths = stock_paths_with_payoff[:, :, date]
        else:
          paths = stock_paths[:, :, date]
      if self.use_rnn:
        h = hs[date]
      else:
        h = None
      stopping_rule, continuation_value = self.stop(
        paths, immediate_exercise_value,
        values, h=h, var_paths=varp,
        return_continuation_values=True)
      which = stopping_rule > 0.5

      M_diff[:, date + 1] = copy.deepcopy(
        np.maximum(payoffs[:, date + 1],prev_cont_val) - continuation_value)

      if date > 0:
        values[which] = immediate_exercise_value[which]
      prev_cont_val = copy.deepcopy(continuation_value)
      prevprev_values = copy.deepcopy(prev_values)
      prev_values = copy.deepcopy(values)
    payoff_0 = payoffs[0, 0]
    lower_bound = max(payoff_0, np.mean(values[self.split:]))
    M = np.cumsum(M_diff, axis=1)
    # print("g(x_0)=", payoff_0, "min p_i=", np.min(values[self.split:]),
    #       "max p_i=", np.max(values[self.split:]))

    if verbose > 0:
      print(np.mean(M_diff[self.split:], axis=0))
    upper_bound = np.mean(np.max(payoffs[self.split:] - M[self.split:], axis=1))

    return lower_bound, upper_bound, time_for_path_gen


  def get_spot_derivative(self, spot, eps, dW, fd_freeze_exe_boundary=True,
                          train_eval_split=2):
    """
    Compute the derivative wrt. the spot (called delta) via the central
    finite difference method, i.e. compute price for original spot value +eps
    and -eps and take the difference divided by 2*eps.
    Possibility to use the same exercise boundary (i.e. the same continuation
    value for the stopping decision), which is implied by the midpoint (original
    value).
    @param spot: original spot value
    @param eps: the epsilon to use
    @param dW: the brownian increments to be reused for all different starting
            values
    @param fd_freeze_exe_boundary: bool, whether to use the same exercise
            boundary for all different starting values
    @return: price (at midpoint), delta, time for path-generation
    """
    t1 = time.time()
    self.model.spot = spot
    stock_paths, var_paths = self.model.generate_paths(dW=dW)
    disc_factor = np.math.exp((-self.model.rate) * self.model.maturity /
                              (self.model.nb_dates))
    self.model.spot = spot + eps
    stock_paths_p, var_paths_p = self.model.generate_paths(dW=dW)
    disc_factor_p = disc_factor
    self.model.spot = spot - eps
    stock_paths_m, var_paths_m = self.model.generate_paths(dW=dW)
    disc_factor_m = disc_factor
    self.model.spot = spot
    time_path_gen = time.time() - t1

    price, delta = self.get_central_derivative(
      stock_paths, var_paths, disc_factor,
      stock_paths_p, var_paths_p, disc_factor_p,
      stock_paths_m, var_paths_m, disc_factor_m,
      eps, fd_freeze_exe_boundary, train_eval_split=train_eval_split)

    return price, delta, time_path_gen

  def get_central_derivative(
          self, stock_paths, var_paths, disc_factor,
          stock_paths_p, var_paths_p, disc_factor_p,
          stock_paths_m, var_paths_m, disc_factor_m,
          eps, fd_freeze_exe_boundary=True, train_eval_split=2):
    """
    compute the midpoint price and the approximation of its derivative via the
    central finite difference method.
    @param stock_paths: paths associated with midpoint of stock-values
    @param var_paths: paths associated with midpoint of var-values, if needed
    @param disc_factor: discount factor associated with midpoint
    @param stock_paths_p: paths associated with midpoint +eps of stock-values
    @param var_paths_p: paths associated with midpoint +eps of var-values, if
            needed
    @param disc_factor_p: discount factor associated with midpoint +eps
    @param stock_paths_m: paths associated with midpoint -eps of stock-values
    @param var_paths_m: paths associated with midpoint -eps of var-values, if
            needed
    @param disc_factor_m: discount factor associated with midpoint -eps
    @param eps: the epsilon value by which midpoint is shifted
    @param fd_freeze_exe_boundary: bool, whether to use exercise boundary of
            midpoint also for +/-eps valeus
    @return: price, derivative
    """
    payoffs = self.payoff(stock_paths)
    stock_paths_with_payoff = np.concatenate(
      [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
    payoffs_p = self.payoff(stock_paths_p)
    stock_paths_with_payoff_p = np.concatenate(
      [stock_paths_p, np.expand_dims(payoffs_p, axis=1)], axis=1)
    payoffs_m = self.payoff(stock_paths_m)
    stock_paths_with_payoff_m = np.concatenate(
      [stock_paths_m, np.expand_dims(payoffs_m, axis=1)], axis=1)
    self.split = int(len(stock_paths)/train_eval_split)
    if self.use_rnn:
      if self.use_payoff_as_input:
        hs = self.compute_hs(stock_paths_with_payoff, var_paths=var_paths)
        hs_p = self.compute_hs(stock_paths_with_payoff_p, var_paths=var_paths_p)
        hs_m = self.compute_hs(stock_paths_with_payoff_m, var_paths=var_paths_m)
      else:
        hs = self.compute_hs(stock_paths, var_paths=var_paths)
        hs_p = self.compute_hs(stock_paths_p, var_paths=var_paths_p)
        hs_m = self.compute_hs(stock_paths_m, var_paths=var_paths_m)

    immediate_exercise_value = self.payoff.eval(stock_paths[:, :, -1])
    values = immediate_exercise_value
    immediate_exercise_value_p = self.payoff.eval(stock_paths_p[:, :, -1])
    values_p = immediate_exercise_value_p
    immediate_exercise_value_m = self.payoff.eval(stock_paths_m[:, :, -1])
    values_m = immediate_exercise_value_m
    for i, date in enumerate(range(stock_paths.shape[2] - 2, 0, -1)):
      self.which_weight = i
      immediate_exercise_value = self.payoff.eval(stock_paths[:, :, date])
      immediate_exercise_value_p = self.payoff.eval(stock_paths_p[:, :, date])
      immediate_exercise_value_m = self.payoff.eval(stock_paths_m[:, :, date])
      if self.use_rnn:
        h = hs[date]
        h_p = hs_p[date]
        h_m = hs_m[date]
      else:
        h, h_p, h_m = None, None, None
      if self.use_path:
        varp, varp_m, varp_p = None, None, None
        if self.use_var:
          varp = var_paths[:, :, :date+1]
          varp_p = var_paths_p[:, :, :date+1]
          varp_m = var_paths_m[:, :, :date+1]
        if self.use_payoff_as_input:
          paths = stock_paths_with_payoff[:, :, :date+1]
          paths_p = stock_paths_with_payoff_p[:, :, :date+1]
          paths_m = stock_paths_with_payoff_m[:, :, :date+1]
        else:
          paths = stock_paths[:, :, :date+1]
          paths_p = stock_paths_p[:, :, :date+1]
          paths_m = stock_paths_m[:, :, :date+1]
      else:
        varp, varp_m, varp_p = None, None, None
        if self.use_var:
          varp = var_paths[:, :, date]
          varp_p = var_paths_p[:, :, date]
          varp_m = var_paths_m[:, :, date]
        if self.use_payoff_as_input:
          paths = stock_paths_with_payoff[:, :, date]
          paths_p = stock_paths_with_payoff_p[:, :, date]
          paths_m = stock_paths_with_payoff_m[:, :, date]
        else:
          paths = stock_paths[:, :, date]
          paths_p = stock_paths_p[:, :, date]
          paths_m = stock_paths_m[:, :, date]

      stopping_rule, continuation_values = self.stop(
        paths, immediate_exercise_value,
        values*disc_factor, h=h, var_paths=varp,
        return_continuation_values=True)
      if not fd_freeze_exe_boundary:
        stopping_rule_p = self.stop(
          paths_p, immediate_exercise_value_p,
          values_p * disc_factor_p, h=h_p, var_paths=varp_p)
        stopping_rule_m = self.stop(
          paths_m, immediate_exercise_value_m,
          values_m * disc_factor_m, h=h_m, var_paths=varp_m)
      else:
        try:
          stopping_rule_p = np.zeros(len(paths))
          if self.train_ITM_only:
            which = (immediate_exercise_value_p > continuation_values) & \
                    (immediate_exercise_value_p > np.finfo(float).eps)
          else:
            which = immediate_exercise_value_p > continuation_values
          stopping_rule_p[which] = 1
          stopping_rule_m = np.zeros(len(paths))
          if self.train_ITM_only:
            which = (immediate_exercise_value_m > continuation_values) & \
                    (immediate_exercise_value_m > np.finfo(float).eps)
          else:
            which = immediate_exercise_value_m > continuation_values
          stopping_rule_m[which] = 1
        except Exception:  # DOS case
          stopping_rule_p = self.stop(
            paths_p, immediate_exercise_value_p,
            values_p * disc_factor_p, h=h_p, var_paths=varp_p,
            train=False, return_continuation_values=False)
          stopping_rule_m = self.stop(
            paths_m, immediate_exercise_value_m,
            values_m * disc_factor_m, h=h_m, var_paths=varp_m,
            train=False, return_continuation_values=False)

      which = stopping_rule > 0.5
      which_p = stopping_rule_p > 0.5
      which_m = stopping_rule_m > 0.5
      values[which] = immediate_exercise_value[which]
      values[~which] *= disc_factor
      values_p[which_p] = immediate_exercise_value_p[which_p]
      values_p[~which_p] *= disc_factor_p
      values_m[which_m] = immediate_exercise_value_m[which_m]
      values_m[~which_m] *= disc_factor_m

    values *= disc_factor
    values_p *= disc_factor_p
    values_m *= disc_factor_m

    price = [np.mean(values[self.split:]), np.mean(values_p[self.split:]),
             np.mean(values_m[self.split:])]
    derivative = np.mean((values_p[self.split:] - values_m[self.split:])/(2*eps))

    return price, derivative

  def get_time_derivative(self, eps, dW, fd_freeze_exe_boundary=False,
                          train_eval_split=2):
    """
    Compute the derivative wrt. the time (called theta) via the central
    finite difference method, i.e. compute price for original maturity value
    -eps (<=> current time +eps) and +eps (<=> current time -eps) and
    take the difference divided by 2*eps.
    Possibility to use the same exercise boundary (i.e. the same continuation
    value for the stopping decision), which is implied by the midpoint (original
    value).
    @param eps: the epsilon to use
    @param dW: the brownian increments to be reused for all different starting
            values
    @param fd_freeze_exe_boundary: bool, whether to use the same exercise
            boundary for all different starting values
    @return: price (at midpoint), theta, time for path-generation
    """
    t1 = time.time()
    stock_paths, var_paths = self.model.generate_paths(dW=dW)
    disc_factor = np.math.exp((-self.model.rate) * self.model.maturity /
                              self.model.nb_dates)
    maturity_old = copy.copy(self.model.maturity)
    self.model.maturity -= eps
    dt_old = copy.copy(self.model.dt)
    self.model.dt = self.model.maturity / self.model.nb_dates
    stock_paths_p, var_paths_p = self.model.generate_paths(
      dW=dW*np.sqrt(self.model.dt)/np.sqrt(dt_old))
    disc_factor_p = np.math.exp((-self.model.rate) * self.model.maturity /
                                self.model.nb_dates)
    self.model.maturity = maturity_old + eps
    self.model.dt = self.model.maturity / self.model.nb_dates
    stock_paths_m, var_paths_m = self.model.generate_paths(
      dW=dW*np.sqrt(self.model.dt)/np.sqrt(dt_old))
    disc_factor_m = np.math.exp((-self.model.rate) * self.model.maturity /
                                self.model.nb_dates)
    self.model.maturity = maturity_old
    self.model.dt = dt_old
    time_path_gen = time.time() - t1

    price, theta = self.get_central_derivative(
      stock_paths, var_paths, disc_factor,
      stock_paths_p, var_paths_p, disc_factor_p,
      stock_paths_m, var_paths_m, disc_factor_m,
      eps, fd_freeze_exe_boundary, train_eval_split=train_eval_split)

    return price, theta, time_path_gen

  def get_rate_derivative(self, eps, dW, fd_freeze_exe_boundary=False,
                          train_eval_split=2):
    """
    Compute the derivative wrt. the rate (called rho) via the central
    finite difference method, i.e. compute price for original rate value
    +eps and -eps and take the difference divided by 2*eps.
    Possibility to use the same exercise boundary (i.e. the same continuation
    value for the stopping decision), which is implied by the midpoint (original
    value).
    @param eps: the epsilon to use
    @param dW: the brownian increments to be reused for all different starting
            values
    @param fd_freeze_exe_boundary: bool, whether to use the same exercise
            boundary for all different starting values
    @return: price (at midpoint), rho, time for path-generation
    """
    t1 = time.time()
    stock_paths, var_paths = self.model.generate_paths(dW=dW)
    disc_factor = np.math.exp((-self.model.rate) * self.model.maturity /
                              self.model.nb_dates)
    rate_old = copy.copy(self.model.rate)
    drift_old = copy.copy(self.model.drift)
    self.model.rate += eps
    self.model.drift += eps
    stock_paths_p, var_paths_p = self.model.generate_paths(dW=dW)
    disc_factor_p = np.math.exp((-self.model.rate) * self.model.maturity /
                                self.model.nb_dates)
    self.model.rate = rate_old - eps
    self.model.drift = drift_old - eps
    stock_paths_m, var_paths_m = self.model.generate_paths(dW=dW)
    disc_factor_m = np.math.exp((-self.model.rate) * self.model.maturity /
                                self.model.nb_dates)
    self.model.rate = rate_old
    self.model.drift = drift_old
    time_path_gen = time.time() - t1

    price, rho = self.get_central_derivative(
      stock_paths, var_paths, disc_factor,
      stock_paths_p, var_paths_p, disc_factor_p,
      stock_paths_m, var_paths_m, disc_factor_m,
      eps, fd_freeze_exe_boundary, train_eval_split=train_eval_split)

    return price, rho, time_path_gen

  def get_vola_derivative(self, eps, dW, fd_freeze_exe_boundary=False,
                          train_eval_split=2):
    """
    Compute the derivative wrt. the volatility (called vega) via the central
    finite difference method, i.e. compute price for original sigma value
    +eps and -eps and take the difference divided by 2*eps.
    Possibility to use the same exercise boundary (i.e. the same continuation
    value for the stopping decision), which is implied by the midpoint (original
    value).
    @param eps: the epsilon to use
    @param dW: the brownian increments to be reused for all different starting
            values
    @param fd_freeze_exe_boundary: bool, whether to use the same exercise
            boundary for all different starting values
    @return: price (at midpoint), vega, time for path-generation
    """
    t1 = time.time()
    stock_paths, var_paths = self.model.generate_paths(dW=dW)
    disc_factor = np.math.exp((-self.model.rate) * self.model.maturity /
                              self.model.nb_dates)
    vola_old = copy.copy(self.model.volatility)
    self.model.volatility += eps
    stock_paths_p, var_paths_p = self.model.generate_paths(dW=dW)
    disc_factor_p = np.math.exp((-self.model.rate) * self.model.maturity /
                                self.model.nb_dates)
    self.model.volatility = vola_old - eps
    stock_paths_m, var_paths_m = self.model.generate_paths(dW=dW)
    disc_factor_m = np.math.exp((-self.model.rate) * self.model.maturity /
                                self.model.nb_dates)
    self.model.volatility = vola_old
    time_path_gen = time.time() - t1

    price, vega = self.get_central_derivative(
      stock_paths, var_paths, disc_factor,
      stock_paths_p, var_paths_p, disc_factor_p,
      stock_paths_m, var_paths_m, disc_factor_m,
      eps, fd_freeze_exe_boundary, train_eval_split=train_eval_split)

    return price, vega, time_path_gen

  def compute_gamma_via_PDE(self, price, delta, theta):
    """
    use the Black-Scholes PDE (possibly with dividend, see:
    https://www.math.tamu.edu/~mike.stecher/425/Sp12/optionsForDividendStocks.pdf)
    to compute the value of gamma out of the price, delta and theta (& rate,
    vola, spot, dividend). only works if the underlying model is a
    1-dim Black-Scholes model, otherwise returns None.
    @param price:
    @param delta:
    @param theta:
    @return: gamma
    """
    if self.model.name == "BlackScholes":
      return utilities.compute_gamma_via_BS_PDE(
        price=price, delta=delta, theta=theta, rate=self.model.rate,
        vola=self.model.volatility, spot=self.model.spot,
        dividend=self.model.dividend)
    return None

  def get_regression(self, spot, eps, d, dW):
    """
    compute the price, delta and gamma via regression method as proposed in:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3503889
    in particular, instead of starting all MC paths at same spot price, use
    random starting points ~N(spot,eps^2) and fit a polynomial regression model
    mapping starting points to prices. price is computed by evaluating regresion
    model at spot price, delta and gamma by computing 1st and 2nd derivative of
    regression model (i.e. computing derivatives of polynomial basis) and
    evaluating them at spot price.
    @param spot: original spot price
    @param eps: std for normal distribution for spot distortion
    @param d: degree of polynomial basis for regression
    @param dW: Brownian increments for path generation
    @return: price, delta, gamma, time for path-generation
    """
    t1 = time.time()
    self.model.spot = spot
    X0 = np.random.normal(loc=spot, scale=eps,
                          size=(self.model.nb_paths,1))
    X0 = np.repeat(X0, repeats=self.model.nb_stocks, axis=1)
    stock_paths, var_paths = self.model.generate_paths(dW=dW, X0=X0)
    time_path_gen = time.time() - t1

    payoffs = self.payoff(stock_paths)
    stock_paths_with_payoff = np.concatenate(
      [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
    self.split = len(stock_paths)
    if self.use_rnn:
      if self.use_payoff_as_input:
        hs = self.compute_hs(stock_paths_with_payoff, var_paths=var_paths)
      else:
        hs = self.compute_hs(stock_paths, var_paths=var_paths)
    disc_factor = np.math.exp((-self.model.rate) * self.model.maturity /
                              (self.model.nb_dates))
    immediate_exercise_value = self.payoff.eval(stock_paths[:, :, -1])
    values = immediate_exercise_value
    for i, date in enumerate(range(stock_paths.shape[2] - 2, -1, -1)):
      self.which_weight = i
      immediate_exercise_value = self.payoff.eval(stock_paths[:, :, date])
      if self.use_rnn:
        h = hs[date]
      else:
        h = None
      if self.use_path:
        varp = None
        if self.use_var:
          varp = var_paths[:, :, :date+1]
        if self.use_payoff_as_input:
          paths = stock_paths_with_payoff[:, :, :date+1]
        else:
          paths = stock_paths[:, :, :date+1]
      else:
        varp = None
        if self.use_var:
          varp = var_paths[:, :, date]
        if self.use_payoff_as_input:
          paths = stock_paths_with_payoff[:, :, date]
        else:
          paths = stock_paths[:, :, date]

      stopping_rule, continuation_values = self.stop(
        paths, immediate_exercise_value,
        values*disc_factor, h=h, var_paths=varp,
        return_continuation_values=True)
      which = stopping_rule > 0.5
      values[which] = immediate_exercise_value[which]
      values[~which] *= disc_factor

    # fit regression to values
    b, b_d, b_g = utilities.get_poly_basis_and_derivatives(X=X0[:, :1], d=d)
    b_val, b_d_val, b_g_val = utilities.get_poly_basis_and_derivatives(
      X=np.array([[spot]]), d=d)
    linreg = LinearRegression(fit_intercept=False)
    res = linreg.fit(X=b, y=values)

    price = linreg.predict(X=b_val)[0]
    delta = linreg.predict(X=b_d_val)[0]
    gamma = linreg.predict(X=b_g_val)[0]

    return price, delta, gamma, time_path_gen

  def price_and_greeks(
          self, eps=0.01, greeks_method="central", fd_freeze_exe_boundary=True,
          poly_deg=2, reg_eps=5, fd_compute_gamma_via_PDE=True,
          train_eval_split=2):
    """
    Computes the price of an American Option using backward recusrion.
    Additionally computes the Greeks: Delta, Gamma, Theta, Rho, Vega via finite
    difference method (or PDE method for gamma; or regression method for price,
    delta, gamma).
    """
    orig_spot = copy.copy(self.model.spot)
    t = time.time()
    if configs.path_gen_seed.get_seed() is not None:
      np.random.seed(configs.path_gen_seed.get_seed())
    stock_paths, var_paths, dW = self.model.generate_paths(return_dW=True)
    t = time.time() - t
    if greeks_method == "central":
      p, delta, t1 = self.get_spot_derivative(
        spot=orig_spot, eps=eps/2, dW=dW,
        fd_freeze_exe_boundary=fd_freeze_exe_boundary,
        train_eval_split=train_eval_split)
      price = p[0]
    elif greeks_method == "forward":
      p, delta, t1 = self.get_spot_derivative(
        spot=orig_spot+eps/2, eps=eps/2, dW=dW,
        fd_freeze_exe_boundary=fd_freeze_exe_boundary,
        train_eval_split=train_eval_split)
      price = p[2]
    elif greeks_method == "backward":
      p, delta, t1 = self.get_spot_derivative(
        spot=orig_spot-eps/2, eps=eps/2, dW=dW,
        fd_freeze_exe_boundary=fd_freeze_exe_boundary,
        train_eval_split=train_eval_split)
      price = p[1]
    elif greeks_method == "regression":
      price, delta, gamma, t1 = self.get_regression(
        spot=orig_spot, eps=reg_eps, d=poly_deg, dW=dW)
    else:
      raise NotImplementedError
    _, theta, t4 = self.get_time_derivative(
      eps=eps/2, dW=dW, fd_freeze_exe_boundary=fd_freeze_exe_boundary,
      train_eval_split=train_eval_split)
    _, rho, t5 = self.get_rate_derivative(
      eps=eps/2, dW=dW, fd_freeze_exe_boundary=fd_freeze_exe_boundary,
      train_eval_split=train_eval_split)
    _, vega, t6 = self.get_vola_derivative(
      eps=eps/2, dW=dW, fd_freeze_exe_boundary=fd_freeze_exe_boundary,
      train_eval_split=train_eval_split)
    if greeks_method == "regression":
      pass
    if fd_compute_gamma_via_PDE:
      gamma = self.compute_gamma_via_PDE(price, delta, theta)
    else:
      gamma = (p[1] - 2* p[0] + p[2]) / ((eps/2)**2)
    return price, t+t1+t4+t5+t6, delta, gamma, theta, rho, vega



class EuropeanOptionPricer:
  """Computes the price of an American Option using backward recusrion.
  """
  def __init__(self, model, payoff, **kwargs):

    #class model: The stochastic process model of the stock (e.g. Black Scholes).
    self.model = model

    #class payoff: The payoff function of the option (e.g. Max call).
    self.payoff = payoff

  def price(self, train_eval_split=2):
    """It computes the price of the European Option using Monte Carlo.
    """
    model = self.model
    t1 = time.time()
    if configs.path_gen_seed.get_seed() is not None:
      np.random.seed(configs.path_gen_seed.get_seed())
    stock_paths, var_paths = self.model.generate_paths()
    payoffs = self.payoff(stock_paths)
    time_for_path_gen = time.time() - t1
    self.split = int(len(stock_paths)/train_eval_split)
    print("time path gen: {}".format(time_for_path_gen), end=" ")
    disc_factor = np.math.exp((-model.rate) * model.maturity /model.nb_dates)

    return np.mean(payoffs[:,-1]) * disc_factor**model.nb_dates, \
           time_for_path_gen


