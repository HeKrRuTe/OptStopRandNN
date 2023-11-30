"""Base class that computes the European and American option price using
finite difference method.

All algorithms that are using a finite difference method such as
B (Binomial Tree)
Trinomial (Trinomial Tree)
PDE (Partial Differential Equation)
"""

from math import exp, sqrt

import numpy as np
import tqdm
import copy
import optimal_stopping.payoffs.payoff as po

import optimal_stopping.algorithms.utils.utilities as utilities



class Finite_Difference_Pricer:
  def __init__(self, model, payoff, **kwargs):
    self.model = model
    self.payoff = payoff
    self.set_vol_and_div()

  def price(self, stock_paths=None, verbose=1, **kwargs):
    raise NotImplementedError

  def set_vol_and_div(self):
    if self.model.nb_stocks > 1 and isinstance(self.payoff, po.Put1Dim):
      self.vol = self.model.volatility/np.sqrt(self.model.nb_stocks)
      self.dividend = self.model.dividend + self.model.volatility**2/2 \
                      - self.vol**2/2
    elif self.model.nb_stocks == 1:
      self.vol = self.model.volatility
      self.dividend = self.model.dividend
    else:
      raise ValueError("Finite differences methods are not implemented for " \
                       "nb_stocks>1 if the payoff is not Put")

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
    self.model.maturity = maturity_old - eps
    price_p, _ = self.price()
    self.model.maturity = maturity_old + eps
    price_m, _ = self.price()
    self.model.maturity = maturity_old
    theta = (price_p - price_m)/(2*eps)
    return price, theta

  def get_rate_derivative(self, eps, price=None):
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
    self.model.volatility = vola_old + eps
    price_p, _ = self.price()
    self.model.volatility = vola_old - eps
    price_m, _ = self.price()
    self.model.volatility = vola_old
    vega = (price_p - price_m)/(2*eps)
    return price, vega

  def compute_gamma_via_PDE(self, price, delta, theta):
    if self.model.name == "BlackScholes":
      vol = self.model.volatility
      nb_stocks = self.model.nb_stocks
      payoff_fct = self.payoff
      dividend = self.model.dividend
      if nb_stocks > 1 and isinstance(payoff_fct, po.Put1Dim):
        vol_hat = vol/np.sqrt(nb_stocks)
        dividend_hat = dividend + vol**2/2 - vol_hat**2/2
      elif nb_stocks == 1:
        vol_hat = vol
        dividend_hat = dividend
      else:
        raise ValueError("Binomial Pricer is not implemented for nb_stocks>1 "
                         "if the payoff is not Put")
      return utilities.compute_gamma_via_BS_PDE(
        price=price, delta=delta, theta=theta, rate=self.model.rate,
        vola=vol_hat, spot=self.model.spot,
        dividend=dividend_hat)
    return None

  def price_and_greeks(
          self, eps=0.01, greeks_method="central",
          fd_compute_gamma_via_PDE=True, **kwargs):
    orig_spot = copy.copy(self.model.spot)
    if greeks_method == "central":
      price, delta, p_p, p_m = self.get_central_derivative(
        spot=orig_spot, eps=eps/2, return_prices_pm=True)
    elif greeks_method == "forward":
      price, delta, p_p, p_m = self.get_central_derivative(
        spot=orig_spot+eps/2, eps=eps/2, return_prices_pm=True)
    elif greeks_method == "backward":
      price, delta, p_p, p_m = self.get_central_derivative(
        spot=orig_spot-eps/2, eps=eps/2, return_prices_pm=True)
    else:
      raise NotImplementedError
    _, theta = self.get_time_derivative(eps=eps/2, price=price)
    _, rho = self.get_rate_derivative(eps=eps/2, price=price)
    _, vega = self.get_vola_derivative(eps=eps/2, price=price)
    if not fd_compute_gamma_via_PDE:
      gamma = (p_p - 2*price + p_m) / ((eps/2)**2)
    else:
      gamma = self.compute_gamma_via_PDE(price, delta, theta)
    return price, 0, delta, gamma, theta, rho, vega
