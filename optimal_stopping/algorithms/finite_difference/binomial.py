from math import exp, sqrt
import numpy as np
import tqdm
from optimal_stopping.algorithms.finite_difference import \
  finite_difference_pricer


class BinomialPricer(finite_difference_pricer.Finite_Difference_Pricer):
  def __init__(self, model, payoff, **kwargs):
    super().__init__(model, payoff, **kwargs)

  def price(self, stock_paths=None, verbose=1):
    self.set_vol_and_div()
    deltaT = self.model.maturity / self.model.nb_dates
    discount_factor = exp(-self.model.rate * deltaT)

    up = exp(self.vol * sqrt(deltaT))
    down = 1 / up

    proba_up = (exp((self.model.rate - self.dividend) * deltaT) - down) / (up - down)
    proba_down = 1 - proba_up

    steps = range(self.model.nb_dates)
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
    i = np.arange(self.model.nb_dates)
    spot_prices = self.model.spot * up**(self.model.nb_dates-1-i) * down**i
    option_prices = self.payoff(spot_prices)
    for n in tqdm.tqdm(reversed(steps[:-1]), disable=(not verbose)):
      spot_prices = spot_prices[:-1]*down
      option_prices = discount_factor * (proba_up * option_prices[:n+1] +
                                         proba_down * option_prices[1:n+2])
      exercise = self.payoff(spot_prices)
      which = option_prices < exercise
      option_prices[which] = exercise[which]
    return option_prices[0], 0
