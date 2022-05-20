import numpy as np

class Payoff:
  def eval(self, X):
    raise NotImplemented()


class MaxPut(Payoff):
  def __init__(self, strike):
    self.strike =  strike

  def __call__(self, X, strike=None):
    assert strike is None or strike == self.strike
    return self.eval(X)

  def eval(self, X):
    # print('payoff.eval ', X, type(X))
    payoff = self.strike - np.max(X, axis=1)
    return payoff.clip(0, None)


class MaxCall(Payoff):
  def __init__(self, strike):
    self.strike =  strike

  def __call__(self, X, strike=None):
    assert strike is None or strike == self.strike
    return self.eval(X)

  def eval(self, X):
    # print('payoff.eval ', X, type(X))
    payoff = np.max(X, axis=1) - self.strike
    return payoff.clip(0, None)


class Put1Dim(Payoff):
  def __init__(self, strike):
    self.strike =  strike

  def __call__(self, X, strike=None):
    assert strike is None or strike == self.strike
    return self.eval(X)

  def eval(self, X):
    return np.maximum(0, self.strike - X)


class Call1Dim(Payoff):
  def __init__(self, strike):
    self.strike =  strike

  def __call__(self, X, strike=None):
    assert strike is None or strike == self.strike
    return self.eval(X)

  def eval(self, X):
    return np.maximum(0, X - self.strike)


class MinPut(Payoff):
  def __init__(self, strike):
    self.strike =  strike

  def __call__(self, X, strike=None):
    assert strike is None or strike == self.strike
    return self.eval(X)

  def eval(self, X):
    payoff = self.strike - np.min(X, axis=1)
    return payoff.clip(0, None)


class GeometricPut(Payoff):
  def __init__(self, strike):
    self.strike = strike

  def __call__(self, X, strike=None):
    assert strike is None or strike == self.strike
    return self.eval(X)

  def eval(self, X):
    dim = len(X[1])  # Here was a mistake, X[0]
    payoff = self.strike - np.prod(X, axis=1) ** (1/dim)
    return payoff.clip(0, None)


class BasketCall(Payoff):
  def __init__(self, strike):
    self.strike = strike

  def __call__(self, X, strike=None):
    assert strike is None or strike == self.strike
    return self.eval(X)

  def eval(self, X):
    payoff = np.mean(X, axis=1) - self.strike
    return payoff.clip(0, None)


class Identity(Payoff):
 def __init__(self, strike):
   self.strike =  strike

 def __call__(self, X, strike=None):
    # assert if it is not a number but a vector
   return self.eval(X)

 def eval(self, X):
   return X[:, 0]


class Max(Payoff):
 def __init__(self, strike):
   self.strike = strike

 def __call__(self, X, strike=None):
   # assert if it is not a number but a vector
   return self.eval(X)

 def eval(self, X):
   return np.max(X, axis=1)


class Mean(Payoff):
 def __init__(self, strike):
   self.strike = strike

 def __call__(self, X, strike=None):
   # assert if it is not a number but a vector
   return self.eval(X)

 def eval(self, X):
   return np.mean(X, axis=1)