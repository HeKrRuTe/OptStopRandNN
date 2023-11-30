from math import exp, sqrt
import numpy as np
import tqdm
from optimal_stopping.algorithms.finite_difference import \
  finite_difference_pricer


class BinomialPricer(finite_difference_pricer.Finite_Difference_Pricer):
  def __init__(self, model, payoff, **kwargs):
    super().__init__(model, payoff, **kwargs)

  def price(self, stock_paths=None, verbose=1, **kwargs):
    self.set_vol_and_div()
    deltaT = self.model.maturity / self.model.nb_dates
    discount_factor = exp(-self.model.rate * deltaT)

    up = exp(self.vol * sqrt(deltaT))
    down = 1 / up

    proba_up = (exp((self.model.rate - self.dividend) * deltaT) - down) / (up - down)
    proba_down = 1 - proba_up

    steps = range(self.model.nb_dates)

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
