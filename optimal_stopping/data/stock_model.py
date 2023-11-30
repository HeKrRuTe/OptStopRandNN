""" Underlying model of the stochastic processes that are used:
- Black Scholes
- Heston
- Fractional Brownian motion
"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
import scipy.special as scispe

import joblib


NB_JOBS_PATH_GEN = 1

class Model:
  def __init__(self, drift, dividend, volatility, spot, nb_stocks,
               nb_paths, nb_dates, maturity, name, **keywords):
    self.name = name
    self.drift = drift - dividend
    self.rate = drift
    self.dividend = dividend
    self.volatility = volatility
    self.spot = spot
    self.nb_stocks = nb_stocks
    self.nb_paths = nb_paths
    self.nb_dates = nb_dates
    self.maturity = maturity
    self.dt = self.maturity / self.nb_dates
    self.df = math.exp(-self.rate * self.dt)
    self.return_var = False

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
    """Returns a nparray (nb_paths * nb_stocks * nb_dates+1) with prices."""
    nb_paths = nb_paths or self.nb_paths
    if NB_JOBS_PATH_GEN > 1:
        return np.array(
            joblib.Parallel(n_jobs=NB_JOBS_PATH_GEN, prefer="threads")(
                joblib.delayed(self.generate_one_path)()
                for i in range(nb_paths)))
    else:
        return np.array([self.generate_one_path() for i in range(nb_paths)]), \
               None


class BlackScholes(Model):
  def __init__(self, drift, volatility, nb_paths, nb_stocks, nb_dates, spot,
         maturity, dividend=0, **keywords):
    super(BlackScholes, self).__init__(
        drift=drift, dividend=dividend, volatility=volatility,
        nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
        spot=spot, maturity=maturity, name="BlackScholes")

  def drift_fct(self, x, t):
    del t
    return self.drift * x

  def diffusion_fct(self, x, t, v=0):
    del t
    return self.volatility * x

  def generate_paths(self, nb_paths=None, return_dW=False, dW=None, X0=None,
                     nb_dates=None):
    """Returns a nparray (nb_paths * nb_stocks * nb_dates) with prices."""
    nb_paths = nb_paths or self.nb_paths
    nb_dates = nb_dates or self.nb_dates
    spot_paths = np.empty((nb_paths, self.nb_stocks, nb_dates+1))
    if X0 is None:
        spot_paths[:, :, 0] = self.spot
    else:
        spot_paths[:, :, 0] = X0
    if dW is None:
        random_numbers = np.random.normal(
            0, 1, (nb_paths, self.nb_stocks, nb_dates))
        dW = random_numbers * np.sqrt(self.dt)
    drift = self.drift
    r = np.repeat(np.repeat(np.repeat(
        np.reshape(drift, (-1, 1, 1)), nb_paths, axis=0),
        self.nb_stocks, axis=1), nb_dates, axis=2)
    sig = np.repeat(np.repeat(np.repeat(
        np.reshape(self.volatility, (-1, 1, 1)), nb_paths, axis=0),
        self.nb_stocks, axis=1), nb_dates, axis=2)
    spot_paths[:, :,  1:] = np.repeat(
        spot_paths[:, :, 0:1], nb_dates, axis=2) * np.exp(np.cumsum(
        r * self.dt - (sig ** 2) * self.dt / 2 + sig * dW, axis=2))
    # dimensions: [nb_paths, nb_stocks, nb_dates+1]
    if return_dW:
        return spot_paths, None, dW
    return spot_paths, None

  def generate_paths_with_alternatives(
          self, nb_paths=None, nb_alternatives=1, nb_dates=None):
    """Returns a nparray (nb_paths * nb_stocks * nb_dates) with prices."""
    nb_paths = nb_paths or self.nb_paths
    nb_dates = nb_dates or self.nb_dates
    total_nb_paths = nb_paths + nb_paths * nb_alternatives * nb_dates
    spot_paths = np.empty((total_nb_paths, self.nb_stocks, nb_dates+1))
    spot_paths[:, :, 0] = self.spot
    random_numbers = np.random.normal(
        0, 1, (total_nb_paths, self.nb_stocks, nb_dates))
    mult = nb_alternatives * nb_paths
    for i in range(nb_dates-1):
        random_numbers[
            nb_paths+i*mult:nb_paths+(i+1)*mult,:,:nb_dates-i-1] = np.tile(
            random_numbers[:nb_paths, :, :nb_dates-i-1],
            reps=(nb_alternatives, 1, 1))
    dW = random_numbers * np.sqrt(self.dt)
    drift = self.drift
    r = np.repeat(np.repeat(np.repeat(
        np.reshape(drift, (-1, 1, 1)), total_nb_paths, axis=0),
        self.nb_stocks, axis=1), nb_dates, axis=2)
    sig = np.repeat(np.repeat(np.repeat(
        np.reshape(self.volatility, (-1, 1, 1)), total_nb_paths, axis=0),
        self.nb_stocks, axis=1), nb_dates, axis=2)
    spot_paths[:, :,  1:] = np.repeat(
        spot_paths[:, :, 0:1], nb_dates, axis=2) * np.exp(np.cumsum(
        r * self.dt - (sig ** 2) * self.dt / 2 + sig * dW, axis=2))
    # dimensions: [nb_paths, nb_stocks, nb_dates+1]
    return spot_paths, None



class FractionalBlackScholes(Model):
  def __init__(self, drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
         maturity, dividend=0, **keywords):
    super(FractionalBlackScholes, self).__init__(
        drift=drift, dividend=dividend, volatility=volatility,
        nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
        spot=spot, maturity=maturity, name="FractionalBlackScholes"
    )
    self.hurst = hurst
    self.fBM = FBM(n=nb_dates, hurst=self.hurst, length=maturity, method='cholesky')


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


class FBMH1:
    """fractional Brownian Motion for hurst H=1"""
    def __init__(self, n, length):
        self.n = n
        self.length = length

    def fbm(self):
        return np.linspace(0, self.length, self.n+1) * np.random.randn(1)


class FractionalBrownianMotion(Model):
  def __init__(self, drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
         maturity, dividend=0, **keywords):
    super(FractionalBrownianMotion, self).__init__(
        drift=drift, dividend=dividend, volatility=volatility,
        nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
        spot=spot, maturity=maturity, name="FractionalBrownianMotion"
    )
    self.hurst = hurst
    if self.hurst == 1:
        self.fBM = FBMH1(n=nb_dates, length=maturity)
    else:
        self.fBM = FBM(n=nb_dates, hurst=hurst, length=maturity, method='cholesky')
    self._nb_stocks = self.nb_stocks

  def _generate_one_path(self):
    """Returns a nparray (nb_stocks * nb_dates) with prices."""
    path = np.empty((self._nb_stocks, self.nb_dates+1))
    for stock in range(self._nb_stocks):
      path[stock, :] = self.fBM.fbm() + self.spot
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
    return path, None




class Heston(Model):
    """
    the Heston model, see: https://en.wikipedia.org/wiki/Heston_model
    a basic stochastic volatility stock price model
    Diffusion of the stock: dS = mu*S*dt + sqrt(v)*S*dW
    Diffusion of the variance: dv = -k(v-vinf)*dt + sqrt(v)*v*dW
    """
    def __init__(self, drift, volatility, mean, speed, correlation, nb_stocks, nb_paths,
                 nb_dates, spot, maturity, dividend=0., sine_coeff=None, **kwargs):
        super(Heston, self).__init__(
            drift=drift, volatility=volatility, nb_stocks=nb_stocks,
            nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot,  maturity=maturity, dividend=dividend, name="Heston"
        )
        self.mean = mean
        self.speed = speed
        self.correlation = correlation

    def drift_fct(self, x, t):
      del t
      return self.drift * x

    def diffusion_fct(self, x, t, v=0):
      del t
      v_positive = np.maximum(v, 0)
      return np.sqrt(v_positive) * x

    def var_drift_fct(self, x, v):
      v_positive = np.maximum(v, 0)
      return - self.speed * (np.subtract(v_positive,self.mean))

    def var_diffusion_fct(self, x, v):
      v_positive = np.maximum(v, 0)
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
        return paths, var_paths


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


class HestonWithVar(Heston):
    def __init__(self, drift, volatility, mean, speed, correlation, nb_stocks, nb_paths,
                 nb_dates, spot, maturity, dividend=0., sine_coeff=None, **kwargs):
        super(HestonWithVar, self).__init__(
            drift, volatility, mean, speed, correlation, nb_stocks, nb_paths,
            nb_dates, spot, maturity, dividend=dividend, sine_coeff=sine_coeff,
            **kwargs
        )
        self.return_var = True


class RoughHeston(Model):
    """
    the Heston model, see: https://en.wikipedia.org/wiki/Heston_model
    a basic stochastic volatility stock price model, that can be used
    even if Feller condition is not satisfied
    Feller condition: 2*speed*mean > volatility**2
    """
    def __init__(self, drift, volatility, spot,
                 mean, speed, correlation,
                 nb_stocks, nb_paths, nb_dates, maturity,
                 nb_steps_mult=10, v0=None, hurst=0.25, dividend=0., **kwargs):
        super(RoughHeston, self).__init__(
            drift=drift, volatility=volatility, nb_stocks=nb_stocks,
            nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot,  maturity=maturity, dividend=dividend,
            name="RoughHeston")
        self.mean = mean
        self.speed = speed
        self.nb_steps_mult = nb_steps_mult
        self.dt = self.maturity/(self.nb_dates*self.nb_steps_mult)
        self.correlation = correlation
        assert 0 < hurst < 1/2
        self.H = hurst

        if v0 is None:
            self.v0 = self.mean
        else:
            self.v0 = v0

        # if 2*self.speed*self.mean > self.volatility**2:
        #     print("Feller condition satisfied")
        # else:
        #     print("Feller condition not satisfied")

    def get_frac_var(self, vars, dZ, step, la, thet, vol,):
        """
        see formula for variance process in paper "Roughening Heston" or at
        https://github.com/sigurdroemer/rough_heston
        Integration with Euler scheme.
        Args:
            vars: array with previous values of var process
            dZ: array with the BM increments for var process
            step: int > 0, the step of the integral
            la: lambda (see formula)
            thet: theta (see formula)
            vol: v (see formula)

        Returns: next value of fractional var process
        """
        v0 = vars[0]
        times = (self.dt*step - np.linspace(0, self.dt*(step-1), step)) ** \
                (self.H - 0.5)
        if len(vars.shape) == 2:
            times = np.repeat(np.expand_dims(times, 1), vars.shape[1], axis=1)
        int1 = np.sum(times*la*(thet-vars[:step])*self.dt, axis=0)
        int2 = np.sum(times*vol*np.sqrt(vars[:step])*dZ[:step], axis=0)
        v = v0 + (int1+int2)/scispe.gamma(self.H + 0.5)
        return np.maximum(v, 0)

    def _generate_one_path(
            self, mu, la, thet, vol, start_X, nb_steps, v0=None):
        spot_path = np.empty((nb_steps + 1))
        spot_path[0] = start_X
        var_path = np.empty((nb_steps + 1))
        if v0 is None:
            var_path[0] = self.v0
        else:
            var_path[0] = v0

        # Diffusion of the spot: dS = mu*S*dt + sqrt(v)*S*dW
        log_spot_drift = lambda v, t: \
            (mu - 0.5 * np.maximum(v, 0))
        log_spot_diffusion = lambda v: np.sqrt(np.maximum(v, 0))

        normal_numbers_1 = np.random.normal(0, 1, nb_steps)
        normal_numbers_2 = np.random.normal(0, 1, nb_steps)
        dW = normal_numbers_1 * np.sqrt(self.dt)
        dZ = (self.correlation * normal_numbers_1 + np.sqrt(
            1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(self.dt)

        for k in range(1, nb_steps + 1):
            spot_path[k] = np.exp(
                np.log(spot_path[k - 1])
                + log_spot_drift(
                    var_path[k - 1], (k-1)*self.dt) * self.dt
                + log_spot_diffusion(var_path[k - 1]) * dW[k-1]
            )
            var_path[k] = self.get_frac_var(var_path, dZ, k, la, thet, vol)
        return spot_path, var_path

    def _generate_paths(
            self, mu, la, thet, vol, start_X, nb_steps, v0=None, nb_stocks=1):
        spot_path = np.empty(
            (nb_steps + 1, nb_stocks))
        spot_path[0] = start_X
        var_path = np.empty(
            (nb_steps + 1, nb_stocks))
        if v0 is None:
            var_path[0] = self.v0
        else:
            var_path[0] = v0

        # Diffusion of the spot: dS = mu*S*dt + sqrt(v)*S*dW
        log_spot_drift = lambda v, t: \
            (mu - 0.5 * np.maximum(v, 0))
        log_spot_diffusion = lambda v: np.sqrt(np.maximum(v, 0))

        normal_numbers_1 = np.random.normal(0, 1, (nb_steps, nb_stocks))
        normal_numbers_2 = np.random.normal(0, 1, (nb_steps, nb_stocks))
        dW = normal_numbers_1 * np.sqrt(self.dt)
        dZ = (self.correlation * normal_numbers_1 + np.sqrt(
            1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(self.dt)

        for k in range(1, nb_steps + 1):
            spot_path[k] = np.exp(
                np.log(spot_path[k - 1])
                + log_spot_drift(
                    var_path[k - 1], (k-1)*self.dt) * self.dt
                + log_spot_diffusion(var_path[k - 1]) * dW[k-1]
            )
            var_path[k] = self.get_frac_var(var_path, dZ, k, la, thet, vol)
        return spot_path, var_path

    def generate_one_path(self):
        """
        generate paths under P, for each dimension (stock) one path is generated
        :param start_X: array with shape of S0 or None, different starting point
        """
        spot_paths = np.empty((self.nb_stocks, self.nb_dates+1))
        for i in range(self.nb_stocks):
            spot_path, var_path = self._generate_one_path(
                self.drift, self.speed, self.mean, self.volatility,
                start_X=self.spot, nb_steps=self.nb_dates*self.nb_steps_mult)
            spot_paths[i, :] = spot_path[0::self.nb_steps_mult]

        return spot_paths

    def generate_paths(self, nb_paths=None):
        """Returns a nparray (nb_paths, nb_stocks, nb_dates) with prices."""
        nb_paths = nb_paths or self.nb_paths
        spot_paths, var_paths = self._generate_paths(
            self.drift, self.speed, self.mean, self.volatility,
            start_X=self.spot, nb_steps=self.nb_dates*self.nb_steps_mult,
            nb_stocks=self.nb_stocks*nb_paths
        )
        spot_paths = spot_paths[0::self.nb_steps_mult]
        spot_paths = np.reshape(spot_paths,
            (1+self.nb_dates, nb_paths, self.nb_stocks))
        spot_paths = np.transpose(spot_paths, axes=(1,2,0))
        var_paths = var_paths[0::self.nb_steps_mult]
        var_paths = np.reshape(var_paths,
                                (1+self.nb_dates, nb_paths, self.nb_stocks))
        var_paths = np.transpose(var_paths, axes=(1,2,0))
        return spot_paths, var_paths


class RoughHestonWithVar(RoughHeston):
    def __init__(self, drift, volatility, spot,
                 mean, speed, correlation,
                 nb_stocks, nb_paths, nb_dates, maturity,
                 nb_steps_mult=10, v0=None, hurst=0.25, dividend=0., **kwargs):
        super(RoughHestonWithVar, self).__init__(
            drift, volatility, spot,
            mean, speed, correlation,
            nb_stocks, nb_paths, nb_dates, maturity,
            nb_steps_mult=nb_steps_mult, v0=v0, hurst=hurst, dividend=dividend,
            **kwargs
        )
        self.return_var = True


# ==============================================================================
# dict for the supported stock models to get them from their name
STOCK_MODELS = {
    "BlackScholes": BlackScholes,
    'FractionalBlackScholes': FractionalBlackScholes,
    'FractionalBrownianMotion': FractionalBrownianMotion,
    'FractionalBrownianMotionPathDep': FractionalBrownianMotionPathDep,
    "Heston": Heston,
    "RoughHeston": RoughHeston,
    "HestonWithVar": HestonWithVar,
    "RoughHestonWithVar": RoughHestonWithVar,
}
# ==============================================================================


hyperparam_test_stock_models = {
    'drift': 0.2, 'volatility': 0.3, 'mean': 0.5, 'speed': 0.5, 'hurst':0.05,
    'correlation': 0.5, 'nb_paths': 1, 'nb_dates': 100, 'maturity': 1.,
    'nb_stocks':10, 'spot':100}

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
    # draw_stock_model("BlackScholes")
    # draw_stock_model("FractionalBlackScholes")
    # heston = STOCK_MODELS["Heston"](**hyperparam_test_stock_models)
    # heston.draw_path_heston("heston.pdf")

    rHeston = RoughHeston(**hyperparam_test_stock_models)
    t = time.time()
    p = rHeston.generate_paths(1000)
    print("needed time: {}".format(time.time()-t))