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
import copy

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

  def calculate_continuation_value(self):
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

  def price(self):
    """It computes the price of an American Option using a backward recursion.
    """
    model = self.model
    t1 = time.time()
    stock_paths, var_paths = self.model.generate_paths()
    payoffs = self.payoff(stock_paths)
    stock_paths_with_payoff = np.concatenate(
      [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
    time_for_path_gen = time.time() - t1
    self.split = int(len(stock_paths)/2)
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
    for date in range(stock_paths.shape[2] - 2, 0, -1):
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
        stopping_rule = self.stop(
          paths, immediate_exercise_value,
          values * disc_factor, h=h, var_paths=varp)
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

  def get_spot_derivative(self, spot, eps, dW, fd_freeze_exe_boundary=True):
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
      eps, fd_freeze_exe_boundary)

    return price, delta, time_path_gen

  def get_spot_derivative2(self, spot, eps, dW, fd_freeze_exe_boundary=True):
    t1 = time.time()
    self.model.spot = spot
    stock_paths, var_paths = self.model.generate_paths(dW=dW)
    disc_factor = np.math.exp((-self.model.rate) * self.model.maturity /
                              (self.model.nb_dates))
    self.model.spot = spot + eps
    stock_paths_p, var_paths_p = self.model.generate_paths(dW=dW)
    disc_factor_p = disc_factor
    self.model.spot = spot - eps
    self.model.spot = spot
    time_path_gen = time.time() - t1

    price, delta = self.get_forward_derivative(
      eps, stock_paths, var_paths, disc_factor,
      stock_paths_p, var_paths_p, disc_factor_p)

    return price, delta, time_path_gen

  def get_central_derivative(
          self, stock_paths, var_paths, disc_factor,
          stock_paths_p, var_paths_p, disc_factor_p,
          stock_paths_m, var_paths_m, disc_factor_m,
          eps, fd_freeze_exe_boundary=True):
    payoffs = self.payoff(stock_paths)
    stock_paths_with_payoff = np.concatenate(
      [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
    payoffs_p = self.payoff(stock_paths_p)
    stock_paths_with_payoff_p = np.concatenate(
      [stock_paths_p, np.expand_dims(payoffs_p, axis=1)], axis=1)
    payoffs_m = self.payoff(stock_paths_m)
    stock_paths_with_payoff_m = np.concatenate(
      [stock_paths_m, np.expand_dims(payoffs_m, axis=1)], axis=1)
    self.split = int(len(stock_paths)/2)
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
    for date in range(stock_paths.shape[2] - 2, 0, -1):
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

  def get_time_derivative(self, eps, dW):
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
    self.model.maturity = maturity_old
    self.model.dt = dt_old
    time_path_gen = time.time() - t1

    price, theta = self.get_forward_derivative(
      eps, stock_paths, var_paths, disc_factor,
      stock_paths_p, var_paths_p, disc_factor_p)

    return price, theta, time_path_gen

  def get_time_derivative2(self, eps, dW, fd_freeze_exe_boundary=False):
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
      eps, fd_freeze_exe_boundary)

    return price, theta, time_path_gen

  def get_rate_derivative(self, eps, dW):
    t1 = time.time()
    stock_paths, var_paths = self.model.generate_paths(dW=dW)
    disc_factor = np.math.exp((-self.model.rate) * self.model.maturity /
                              self.model.nb_dates)
    rate_old = copy.copy(self.model.rate)
    self.model.rate += eps
    stock_paths_p, var_paths_p = self.model.generate_paths(dW=dW)
    disc_factor_p = np.math.exp((-self.model.rate) * self.model.maturity /
                                self.model.nb_dates)
    self.model.rate = rate_old
    time_path_gen = time.time() - t1

    price, rho = self.get_forward_derivative(
      eps, stock_paths, var_paths, disc_factor,
      stock_paths_p, var_paths_p, disc_factor_p)

    return price, rho, time_path_gen

  def get_vola_derivative(self, eps, dW):
    t1 = time.time()
    stock_paths, var_paths = self.model.generate_paths(dW=dW)
    disc_factor = np.math.exp((-self.model.rate) * self.model.maturity /
                              self.model.nb_dates)
    vola_old = copy.copy(self.model.volatility)
    self.model.volatility += eps
    stock_paths_p, var_paths_p = self.model.generate_paths(dW=dW)
    disc_factor_p = disc_factor
    self.model.volatility = vola_old
    time_path_gen = time.time() - t1

    price, vega = self.get_forward_derivative(
      eps, stock_paths, var_paths, disc_factor,
      stock_paths_p, var_paths_p, disc_factor_p)

    return price, vega, time_path_gen

  def get_forward_derivative(
          self, eps, stock_paths, var_paths, disc_factor,
          stock_paths_p, var_paths_p, disc_factor_p):
    payoffs = self.payoff(stock_paths)
    stock_paths_with_payoff = np.concatenate(
      [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
    payoffs_p = self.payoff(stock_paths_p)
    stock_paths_with_payoff_p = np.concatenate(
      [stock_paths_p, np.expand_dims(payoffs_p, axis=1)], axis=1)
    self.split = int(len(stock_paths)/2)
    if self.use_rnn:
      if self.use_payoff_as_input:
        hs = self.compute_hs(stock_paths_with_payoff, var_paths=var_paths)
        hs_p = self.compute_hs(stock_paths_with_payoff_p, var_paths=var_paths_p)
      else:
        hs = self.compute_hs(stock_paths, var_paths=var_paths)
        hs_p = self.compute_hs(stock_paths_p, var_paths=var_paths_p)
    immediate_exercise_value = self.payoff.eval(stock_paths[:, :, -1])
    values = immediate_exercise_value
    immediate_exercise_value_p = self.payoff.eval(stock_paths_p[:, :, -1])
    values_p = immediate_exercise_value_p
    for date in range(stock_paths.shape[2] - 2, 0, -1):
      immediate_exercise_value = self.payoff.eval(stock_paths[:, :, date])
      immediate_exercise_value_p = self.payoff.eval(stock_paths_p[:, :, date])
      if self.use_rnn:
        h = hs[date]
        h_p = hs_p[date]
      else:
        h, h_p, h_m = None, None, None
      if self.use_path:
        varp, varp_m, varp_p = None, None, None
        if self.use_var:
          varp = var_paths[:, :, :date+1]
          varp_p = var_paths_p[:, :, :date+1]
        if self.use_payoff_as_input:
          paths = stock_paths_with_payoff[:, :, :date+1]
          paths_p = stock_paths_with_payoff_p[:, :, :date+1]
        else:
          paths = stock_paths[:, :, :date+1]
          paths_p = stock_paths_p[:, :, :date+1]
      else:
        varp, varp_m, varp_p = None, None, None
        if self.use_var:
          varp = var_paths[:, :, date]
          varp_p = var_paths_p[:, :, date]
        if self.use_payoff_as_input:
          paths = stock_paths_with_payoff[:, :, date]
          paths_p = stock_paths_with_payoff_p[:, :, date]
        else:
          paths = stock_paths[:, :, date]
          paths_p = stock_paths_p[:, :, date]

      stopping_rule, continuation_values = self.stop(
        paths, immediate_exercise_value,
        values*disc_factor, h=h, var_paths=varp,
        return_continuation_values=True)
      stopping_rule_p = self.stop(
        paths_p, immediate_exercise_value_p,
        values_p * disc_factor_p, h=h_p, var_paths=varp_p)

      which = stopping_rule > 0.5
      which_p = stopping_rule_p > 0.5
      values[which] = immediate_exercise_value[which]
      values[~which] *= disc_factor
      values_p[which_p] = immediate_exercise_value_p[which_p]
      values_p[~which_p] *= disc_factor_p

    values *= disc_factor
    values_p *= disc_factor_p

    price = np.mean(values[self.split:])
    derivative = np.mean((values_p[self.split:] - values[self.split:])/eps)

    return price, derivative

  def compute_gamma_via_PDE(self, price, delta, theta):
    if self.model.name == "BlackScholes":
      return utilities.compute_gamma_via_BS_PDE(
        price=price, delta=delta, theta=theta, rate=self.model.rate,
        vola=self.model.volatility, spot=self.model.spot)
    return None

  def get_regression(self, spot, eps, d, dW):
    t1 = time.time()
    self.model.spot = spot
    X0 = np.random.normal(loc=spot, scale=eps,
                          size=(self.model.nb_paths,self.model.nb_stocks))
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
    for date in range(stock_paths.shape[2] - 2, 0, -1):
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
    values *= disc_factor

    # fit regression to values
    b, b_d, b_g = utilities.get_poly_basis_and_derivatives(X=X0, d=d)
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
          poly_deg=2, fd_compute_gamma_via_PDE=True):
    """
    Computes the price of an American Option using backward recusrion.
    Additionally computes the Delta, Gamma, Theta Greeks via finite difference
    or regression method.
    """
    orig_spot = copy.copy(self.model.spot)
    t = time.time()
    stock_paths, var_paths, dW = self.model.generate_paths(return_dW=True)
    t = time.time() - t
    if greeks_method == "central":
      price, delta, t1 = self.get_spot_derivative(
        spot=orig_spot, eps=eps/2, dW=dW,
        fd_freeze_exe_boundary=fd_freeze_exe_boundary)
      price = price[0]
      if not fd_compute_gamma_via_PDE:
        _, delta1, t2 = self.get_spot_derivative(
          spot=orig_spot+eps/2, eps=eps/2, dW=dW,
          fd_freeze_exe_boundary=fd_freeze_exe_boundary)
        _, delta2, t3 = self.get_spot_derivative(
          spot=orig_spot-eps/2, eps=eps/2, dW=dW,
          fd_freeze_exe_boundary=fd_freeze_exe_boundary)
    elif greeks_method == "forward":
      price, delta, t1 = self.get_spot_derivative(
        spot=orig_spot+eps/2, eps=eps/2, dW=dW,
        fd_freeze_exe_boundary=fd_freeze_exe_boundary)
      price = price[2]
      if not fd_compute_gamma_via_PDE:
        _, delta1, t2 = self.get_spot_derivative(
          spot=orig_spot+3*eps/2, eps=eps/2, dW=dW,
          fd_freeze_exe_boundary=fd_freeze_exe_boundary)
        delta2 = delta
        t3 = 0
    elif greeks_method == "backward":
      price, delta, t1 = self.get_spot_derivative(
        spot=orig_spot-eps/2, eps=eps/2, dW=dW,
        fd_freeze_exe_boundary=fd_freeze_exe_boundary)
      price = price[1]
      if not fd_compute_gamma_via_PDE:
        delta1 = delta
        t2 = 0
        _, delta2, t3 = self.get_spot_derivative(
          spot=orig_spot-3*eps/2, eps=eps/2, dW=dW,
          fd_freeze_exe_boundary=fd_freeze_exe_boundary)
    elif greeks_method == "regression":
      price, delta, gamma, t1 = self.get_regression(
        spot=orig_spot, eps=eps, d=poly_deg, dW=dW)
    else:
      raise NotImplementedError
    _, theta, t4 = self.get_time_derivative(eps=1e-14, dW=dW)
    _, rho, t5 = self.get_rate_derivative(eps=1e-14, dW=dW)
    _, vega, t6 = self.get_vola_derivative(eps=1e-14, dW=dW)
    if greeks_method == "regression":
      return price, t+t1+t4+t5+t6, delta, gamma, theta, rho, vega
    if fd_compute_gamma_via_PDE:
      gamma = self.compute_gamma_via_PDE(price, delta, theta)
      return price, t+t1+t4+t5+t6, delta, gamma, theta, rho, vega
    else:
      gamma = (delta1 - delta2) / eps
      return price, t+t1+t2+t3+t4+t5+t6, delta, gamma, theta, rho, vega



class EuropeanOptionPricer:
  """Computes the price of an American Option using backward recusrion.
  """
  def __init__(self, model, payoff, **kwargs):

    #class model: The stochastic process model of the stock (e.g. Black Scholes).
    self.model = model

    #class payoff: The payoff function of the option (e.g. Max call).
    self.payoff = payoff

  def price(self):
    """It computes the price of the European Option using Monte Carlo.
    """
    model = self.model
    t1 = time.time()
    stock_paths, var_paths = self.model.generate_paths()
    payoffs = self.payoff(stock_paths)
    time_for_path_gen = time.time() - t1
    self.split = int(len(stock_paths)/2)
    print("time path gen: {}".format(time_for_path_gen), end=" ")
    disc_factor = np.math.exp((-model.rate) * model.maturity /model.nb_dates)

    return np.mean(payoffs[:,-1]) * disc_factor**model.nb_dates, \
           time_for_path_gen


