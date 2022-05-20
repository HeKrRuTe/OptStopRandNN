from math import exp, sqrt

import numpy as np
import tqdm
import copy

import optimal_stopping.algorithms.utils.utilities as utilities




def put(spot, strike):
  """Returns put for given spot and strike."""
  return max(strike-spot, 0)

class BinomialPricer:
  def __init__(self, model, payoff, **kwargs):
    self.model = model
    self.payoff = payoff

  def price(self, stock_paths=None, verbose=1):
    maturity = self.model.maturity
    spot = self.model.spot
    vol = self.model.volatility
    nb_dates = self.model.nb_dates
    rate = self.model.rate
    drift = self.model.drift

    payoff_fct = self.payoff

    deltaT = maturity / nb_dates
    discount_factor = exp(-rate * deltaT)
    up = exp(vol * sqrt(deltaT))
    down = 1 / up
    proba_up = (exp(drift * deltaT) - down) / (up - down)
    proba_down = 1 - proba_up
    steps = range(nb_dates)
    #  ---- OLD (SLOW) VERSION ----
    # spot_prices = [[None] * (i+1) for i in steps]
    # option_prices = [[None] * (i+1) for i in steps]
    #
    # # Forward: set the stock prices
    # for n in steps:
    #   for i in range(n+1):
    #     spot_prices[n][i] = spot * up**(n-i) * down**i
    #
    # Backward: compute the option prices
    # First initialize values at maturity ([-1] means last element)
    #
    # option_prices[-1] = [payoff_fct(spot_price) for spot_price in spot_prices[-1]]
    # # Then move to earlier steps
    # for n in tqdm.tqdm(reversed(steps[:-1]), disable=(not verbose)):  # t[:-1] is to remove last element
    #   for i in range(n+1):
    #     option_prices[n][i] = discount_factor * (proba_up * option_prices[n+1][i] + proba_down * option_prices[n+1][i+1])
    #     exercise = payoff_fct(spot_prices[n][i])
    #     if option_prices[n][i] < exercise:
    #       option_prices[n][i] = exercise
    # return option_prices[0][0], 0
    # -----------------------------

    # NEW (FAST) VERSION
    i = np.arange(nb_dates)
    spot_prices = spot * up**(nb_dates-1-i) * down**i
    option_prices = payoff_fct(spot_prices)
    for n in tqdm.tqdm(reversed(steps[:-1]), disable=(not verbose)):
      i = np.arange(n+1)
      spot_prices = spot * up**(n-i) * down**i
      option_prices = discount_factor * (proba_up * option_prices[:n+1] +
                                         proba_down * option_prices[1:n+2])
      exercise = payoff_fct(spot_prices)
      which = option_prices < exercise
      option_prices[which] = exercise[which]
    return option_prices[0], 0

  def get_central_derivative(
          self, spot, eps, compute_price=True, price_p=None, price_m=None,
          return_prices_pm=False):
    price = None
    if compute_price:
      self.model.spot = spot
      price, _ = self.price()
    self.model.spot = spot + eps
    if price_p is None:
      price_p, _ = self.price()
    self.model.spot = spot - eps
    if price_m is None:
      price_m, _ = self.price()
    self.model.spot = spot
    delta = (price_p - price_m)/(2*eps)
    if return_prices_pm:
      return price, delta, price_p, price_m
    return price, delta

  def get_time_derivative(self, eps, price=None):
    maturity_old = copy.copy(self.model.maturity)
    if price is None:
      price, _ = self.price()
    self.model.maturity = maturity_old - eps
    price_p, _ = self.price()
    self.model.maturity = maturity_old
    theta = (price_p - price)/eps
    return price, theta

  def get_time_derivative2(self, eps, price=None):
    maturity_old = copy.copy(self.model.maturity)
    self.model.maturity = maturity_old - eps
    price_p, _ = self.price()
    self.model.maturity = maturity_old + eps
    price_m, _ = self.price()
    self.model.maturity = maturity_old
    theta = (price_p - price_m)/(2*eps)
    return price, theta

  def get_rate_derivative(self, eps, price=None):
    rate_old = copy.copy(self.model.rate)
    if price is None:
      price, _ = self.price()
    self.model.rate = rate_old + eps
    price_p, _ = self.price()
    self.model.rate = rate_old
    rho = (price_p - price)/eps
    return price, rho

  def get_rate_derivative2(self, eps, price=None):
    rate_old = copy.copy(self.model.rate)
    self.model.rate = rate_old + eps
    price_p, _ = self.price()
    self.model.rate = rate_old - eps
    price_m, _ = self.price()
    self.model.rate = rate_old
    rho = (price_p - price_m)/(2*eps)
    return price, rho

  def get_vola_derivative(self, eps, price=None):
    vola_old = copy.copy(self.model.volatility)
    if price is None:
      price, _ = self.price()
    self.model.volatility = vola_old + eps
    price_p, _ = self.price()
    self.model.volatility = vola_old
    vega = (price_p - price)/eps
    return price, vega

  def get_vola_derivative2(self, eps, price=None):
    vola_old = copy.copy(self.model.volatility)
    self.model.volatility = vola_old + eps
    price_p, _ = self.price()
    self.model.volatility = vola_old - eps
    price_m, _ = self.price()
    self.model.volatility = vola_old
    vega = (price_p - price_m)/(2*eps)
    return price, vega

  def compute_gamma_via_PDE(self, price, delta, theta):
    if self.model.name == "BlackScholes":
      return utilities.compute_gamma_via_BS_PDE(
        price=price, delta=delta, theta=theta, rate=self.model.rate,
        vola=self.model.volatility, spot=self.model.spot)
    return None

  def price_and_greeks(
          self, eps=0.01, greeks_method="central",
          fd_compute_gamma_via_PDE=True, **kwargs):
    orig_spot = copy.copy(self.model.spot)
    if greeks_method == "central":
      price, delta = self.get_central_derivative(spot=orig_spot, eps=eps/2)
      if not fd_compute_gamma_via_PDE:
        _, delta1 = self.get_central_derivative(
          spot=orig_spot+eps/2, eps=eps/2, compute_price=False, price_m=price)
        _, delta2 = self.get_central_derivative(
          spot=orig_spot-eps/2, eps=eps/2, compute_price=False, price_p=price)
    elif greeks_method == "forward":
      price, delta = self.get_central_derivative(
        spot=orig_spot+eps/2, eps=eps/2)
      if not fd_compute_gamma_via_PDE:
        _, delta1 = self.get_central_derivative(
          spot=orig_spot+3*eps/2, eps=eps/2, compute_price=False)
        delta2 = delta
    elif greeks_method == "backward":
      price, delta = self.get_central_derivative(
        spot=orig_spot-eps/2, eps=eps/2)
      if not fd_compute_gamma_via_PDE:
        delta1 = delta
        _, delta2 = self.get_central_derivative(
          spot=orig_spot-3*eps/2, eps=eps/2, compute_price=False)
    else:
      raise NotImplementedError
    # _, theta = self.get_time_derivative(eps=eps, price=price)
    # _, rho = self.get_rate_derivative(eps=eps, price=price)
    # _, vega = self.get_vola_derivative(eps=eps, price=price)
    _, theta = self.get_time_derivative2(eps=eps/2, price=price)
    _, rho = self.get_rate_derivative2(eps=eps/2, price=price)
    _, vega = self.get_vola_derivative2(eps=eps/2, price=price)
    if not fd_compute_gamma_via_PDE:
      gamma = (delta1 - delta2) / eps
    else:
      gamma = self.compute_gamma_via_PDE(price, delta, theta)
    return price, 0, delta, gamma, theta, rho, vega



