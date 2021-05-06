""" Underlying model of the stochastic processes that are used:
- Black Scholes
- Heston
- Fractional Brownian motion
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM

import joblib


NB_JOBS_PATH_GEN = 1

class Model:
  def __init__(self, drift, volatility, spot, nb_stocks,  nb_paths, nb_dates,
         maturity, **keywords):
    self.drift = drift
    self.volatility = volatility
    self.spot = spot
    self.nb_stocks = nb_stocks
    self.nb_paths = nb_paths
    self.nb_dates = nb_dates
    self.maturity = maturity
    self.dt = self.maturity / self.nb_dates
    self.df = math.exp(-drift * self.dt)

  def disc_factor(self, date_begin, date_end):
    time = (date_end - date_begin) * self.dt
    return math.exp(-self.drift * time)

  def drift_fct(self, x, t):
    raise NotImplemented()

  def diffusion_fct(self, x, t, v=0):
    raise NotImplemented()

  def generate_one_path(self):
      raise NotImplemented()

  def generate_paths(self, nb_paths=None):
    """Returns a nparray (nb_paths * nb_stocks * nb_dates) with prices."""
    nb_paths = nb_paths or self.nb_paths
    if NB_JOBS_PATH_GEN > 1:
        return np.array(
            joblib.Parallel(n_jobs=NB_JOBS_PATH_GEN, prefer="threads")(
                joblib.delayed(self.generate_one_path)()
                for i in range(nb_paths)))
    else:
        return np.array([self.generate_one_path() for i in range(nb_paths)])


class BlackScholes(Model):
  def __init__(self, drift, volatility, nb_paths, nb_stocks, nb_dates, spot,
         maturity, dividend=0, **keywords):
    super(BlackScholes, self).__init__(drift=drift - dividend, volatility=volatility,
             nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
             spot=spot, maturity=maturity)
    self.drift = drift   # included for dicounting as drift can be != drift

  def drift_fct(self, x, t):
    del t
    return self.drift * x

  def diffusion_fct(self, x, t, v=0):
    del t
    return self.volatility * x

  def generate_one_path(self):
    """Returns a nparray (nb_stocks * nb_dates) with prices."""
    path = np.empty((self.nb_stocks, self.nb_dates+1))
    path[:, 0] = self.spot
    for k in range(1, self.nb_dates+1):
      random_numbers = np.random.normal(0, 1, self.nb_stocks)
      dW = random_numbers*np.sqrt(self.dt)
      previous_spots = path[:, k - 1]
      diffusion = self.diffusion_fct(previous_spots, (k) * self.dt)
      path[:, k] = (
          previous_spots
          + self.drift_fct(previous_spots, (k) * self.dt) * self.dt
          + np.multiply(diffusion, dW))
    return path



class FractionalBlackScholes(BlackScholes):
  def __init__(self, drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
         maturity, dividend=0, **keywords):
    super(FractionalBlackScholes, self).__init__(drift, volatility, nb_paths, nb_stocks, nb_dates, spot, maturity, dividend, **keywords)
    self.drift = drift
    self.hurst = hurst
    self.fBM = FBM(n=nb_dates, hurst=self.hurst, length=maturity, method='hosking')


  def generate_one_path(self):
    """Returns a nparray (nb_stocks * nb_dates) with prices."""
    path = np.empty((self.nb_stocks, self.nb_dates+1))
    fracBM_noise = np.empty((self.nb_stocks, self.nb_dates))
    path[:, 0] = self.spot
    for stock in range(self.nb_stocks):
      fracBM_noise[stock, :] = self.fBM.fgn()
    for k in range(1, self.nb_dates+1):
      previous_spots = path[:, k - 1]
      diffusion = self.diffusion_fct(previous_spots, (k) * self.dt)
      path[:, k] = (
          previous_spots
          + self.drift_fct(previous_spots, (k) * self.dt) * self.dt
          + np.multiply(diffusion, fracBM_noise[:,k-1]))
    print("path",path)
    return path


class FractionalBrownianMotion(BlackScholes):
  def __init__(self, drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
         maturity, dividend=0, **keywords):
    super(FractionalBrownianMotion, self).__init__(drift, volatility, nb_paths, nb_stocks, nb_dates, spot, maturity, dividend, **keywords)
    self.hurst = hurst
    self.fBM = FBM(n=nb_dates, hurst=hurst, length=maturity, method='cholesky')
    self._nb_stocks = self.nb_stocks

  def _generate_one_path(self):
    """Returns a nparray (nb_stocks * nb_dates) with prices."""
    path = np.empty((self._nb_stocks, self.nb_dates+1))
    for stock in range(self._nb_stocks):
      path[stock, :] = self.fBM.fbm() + self.spot
    # print("path",path)
    return path

  def generate_one_path(self):
    return self._generate_one_path()


class FractionalBrownianMotionPathDep(FractionalBrownianMotion):
  def __init__(
          self, drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
          maturity, dividend=0, **keywords):
    assert nb_stocks == 1
    assert spot == 0
    super(FractionalBrownianMotionPathDep, self).__init__(
      drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
      maturity, dividend=0, **keywords)
    self.nb_stocks = nb_dates + 1
    self._nb_stocks = 1

  def generate_one_path(self):
    """Returns a nparray (nb_stocks * nb_dates) with prices."""
    _path = self._generate_one_path()
    path = np.zeros((self.nb_stocks, self.nb_dates+1))
    for i in range(self.nb_dates+1):
      path[:i+1, i] = np.flip(_path[0, :i+1])
    return path




class Heston(Model):
    """
    the Heston model, see: https://en.wikipedia.org/wiki/Heston_model
    a basic stochastic volatility stock price model
    Diffusion of the stock: dS = mu*S*dt + sqrt(v)*S*dW
    Diffusion of the variance: dv = -k(v-vinf)*dt + sqrt(v)*v*dW
    """
    def __init__(self, drift, volatility, mean, speed, correlation, nb_stocks, nb_paths,
                 nb_dates, spot, maturity, sine_coeff=None, **kwargs):
        super(Heston, self).__init__(
            drift=drift, volatility=volatility, nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot,  maturity=maturity,
        )
        self.mean = mean
        self.speed = speed
        self.correlation = correlation

    def drift_fct(self, x, t):
      del t
      return self.drift * x

    def diffusion_fct(self, x, t, v=0):
      del t
      v_positive = [max(i,0.0) for i in v]
      return np.sqrt(v_positive) * x

    def var_drift_fct(self, x, v):
      v_positive = [max(i,0.0) for i in v]
      return - self.speed * (np.subtract(v_positive,self.mean))

    def var_diffusion_fct(self, x, v):
      v_positive = [max(i,0.0) for i in v]
      return self.volatility * np.sqrt(v_positive)

    def generate_paths(self, start_X=None):
        paths = np.empty(
            (self.nb_paths, self.nb_stocks, self.nb_dates + 1))
        var_paths = np.empty(
            (self.nb_paths, self.nb_stocks, self.nb_dates + 1))

        dt = self.maturity / self.nb_dates
        if start_X is not None:
          paths[:, :, 0] = start_X
        for i in range(self.nb_paths):
          if start_X is None:
            paths[i, :, 0] = self.spot
            var_paths[i, :, 0] = self.mean
            for k in range(1, self.nb_dates + 1):
                normal_numbers_1 = np.random.normal(0, 1, self.nb_stocks)
                normal_numbers_2 = np.random.normal(0, 1, self.nb_stocks)
                dW = normal_numbers_1 * np.sqrt(dt)
                dZ = (self.correlation * normal_numbers_1 + np.sqrt(
                    1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(dt)

                var_paths[i, :, k] = (
                        var_paths[i, :, k - 1]
                        + self.var_drift_fct(paths[i, :, k - 1],
                                             var_paths[i, :, k - 1], ) * dt
                        + np.multiply(
                    self.var_diffusion_fct(paths[i, :, k - 1],
                                           var_paths[i, :, k - 1]), dZ))

                paths[i, :, k] = (
                        paths[i, :, k - 1]
                        + self.drift_fct(paths[i, :, k - 1],
                                        (k-1) * dt) * dt
                        + np.multiply(self.diffusion_fct(paths[i, :, k - 1],
                                                    (k) * dt,
                                                    var_paths[i, :, k]), dW))
        return paths


    def draw_path_heston(self, filename):
        nb_paths = self.nb_paths
        self.nb_paths = 1
        paths = self.generate_paths()
        self.nb_paths = nb_paths
        paths, var_paths = paths
        one_path = paths[0, 0, :]
        one_var_path = var_paths[0, 0, :]
        dates = np.array([i for i in range(len(one_path))])
        dt = self.maturity / self.nb_dates
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('time')
        ax1.set_ylabel('Stock', color=color)
        ax1.plot(dates, one_path, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        color = 'tab:red'
        ax2 = ax1.twinx()
        ax2.set_ylabel('Variance', color=color)
        ax2.plot(dates, one_var_path, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.savefig(filename)
        plt.close()




# ==============================================================================
# dict for the supported stock models to get them from their name
STOCK_MODELS = {
    "BlackScholes": BlackScholes,
    'FractionalBlackScholes':FractionalBlackScholes,
    'FractionalBrownianMotion':FractionalBrownianMotion,
    'FractionalBrownianMotionPathDep':FractionalBrownianMotionPathDep,
    "Heston": Heston,
}
# ==============================================================================


hyperparam_test_stock_models = {
    'drift': 0.2, 'volatility': 0.3, 'mean': 0.5, 'speed': 0.5, 'hurst':0.75,
    'correlation': 0.5, 'nb_paths': 1, 'nb_dates': 100, 'maturity': 1., 'nb_stocks':10, 'nb_dates':1000, 'spot':100}

def draw_stock_model(stock_model_name):
    hyperparam_test_stock_models['model_name'] = stock_model_name
    stockmodel = STOCK_MODELS[stock_model_name](**hyperparam_test_stock_models)
    stock_paths = stockmodel.generate_paths()
    filename = '{}.pdf'.format(stock_model_name)

    # draw a path
    one_path = stock_paths[0, 0, :]
    dates = np.array([i for i in range(len(one_path))])
    plt.plot(dates, one_path, label='stock path')
    plt.legend()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    draw_stock_model("BlackScholes")
    draw_stock_model("FractionalBlackScholes")
    heston = STOCK_MODELS["Heston"](**hyperparam_test_stock_models)
    heston.draw_path_heston("heston.pdf")
