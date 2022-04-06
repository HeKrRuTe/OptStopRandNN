import numpy as np
from itertools import combinations


class BasisFunctions:
    def __init__(self, nb_stocks):
        self.nb_stocks = nb_stocks
        lst = list(range(self.nb_stocks))
        self.combs =  [list(x) for x in combinations(lst, 2)]
        self.nb_base_fcts = 1 + 2 * self.nb_stocks + len(self.combs)
        # print("self.nb_base_fcts", self.nb_base_fcts)
        # print("nb_stocks",  self.nb_stocks)

    def base_fct(self, i, x, d2=False):
        if d2:
            bf=np.nan
            if (i == 0):
                bf = np.ones_like(x[:, 0]) # (constant)
            elif (i <= self.nb_stocks):
                bf = x[:, i-1] # (x1, x2, ..., xn)
            elif (self.nb_stocks < i <= 2 * self.nb_stocks):
                k = i - self.nb_stocks - 1
                bf = x[:, k] ** 2 # (x1^2, x2^2, ..., xn^2)
            elif (i > 2 * self.nb_stocks):
                k = i - 2*self.nb_stocks -1
                bf = x[:, self.combs[k][0]] * x[:, self.combs[k][1]] # (x1x2, ..., xn-1xn)
            return bf
        bf=np.nan
        if (i == 0):
            bf = np.ones_like(x[0]) # (constant)
        elif (i <= self.nb_stocks):
            bf = x[i-1] # (x1, x2, ..., xn)
        elif (self.nb_stocks < i <= 2 * self.nb_stocks):
            k = i - self.nb_stocks - 1
            bf = x[k] ** 2 # (x1^2, x2^2, ..., xn^2)
        elif (i > 2 * self.nb_stocks):
            k = i - 2*self.nb_stocks -1
            bf = x[self.combs[k][0]] * x[self.combs[k][1]] # (x1x2, ..., xn-1xn)
        return bf


class BasisFunctionsDeg1:
    def __init__(self, nb_stocks):
        self.nb_stocks = nb_stocks
        self.nb_base_fcts = 1 + self.nb_stocks
        # print("self.nb_base_fcts", self.nb_base_fcts)
        # print("nb_stocks",  self.nb_stocks)

    def base_fct(self, i, x):
        bf=np.nan
        if (i == 0):
            bf = np.ones_like(x[0]) # (constant)
        elif (i <= self.nb_stocks):
            bf = x[i-1] # (x1, x2, ..., xn)
        return bf


class BasisFunctionsLaguerre:
    def __init__(self, nb_stocks, K=1):
        self.nb_stocks = nb_stocks
        self.nb_base_fcts = 1 + 3 * self.nb_stocks
        self.K = K
        # print("self.nb_base_fcts", self.nb_base_fcts)
        # print("nb_stocks",  self.nb_stocks)

    def base_fct(self, i, x):
        bf=np.nan
        x = x / self.K
        if (i == 0):
            bf = np.ones_like(x[0]) # (constant)
        elif (i <= self.nb_stocks):
            bf = np.exp(-x[i-1]/2)
        elif (self.nb_stocks < i <= 2 * self.nb_stocks):
            k = i - self.nb_stocks - 1
            bf = np.exp(-x[k]/2)*(1-x[k])
        elif (i > 2 * self.nb_stocks):
            k = i - 2*self.nb_stocks -1
            bf = np.exp(-x[k]/2)*(1-2*x[k]+(x[k]**2)/2)
        return bf


class BasisFunctionsLaguerreTime:
    """assumes that the last stock is the current time"""
    def __init__(self, nb_stocks, T, K=1):
        self.nb_stocks = nb_stocks
        self.nb_base_fcts = 1 + 3 * self.nb_stocks
        self.T = T
        self.K = K
        # print("self.nb_base_fcts", self.nb_base_fcts)
        # print("nb_stocks",  self.nb_stocks)

    def base_fct(self, i, x):
        bf = np.nan
        x = x / self.K
        if (i == 0):
            bf = np.ones_like(x[0]) # (constant)
        elif (i < self.nb_stocks):
            bf = np.exp(-x[i-1]/2)
        elif i == self.nb_stocks:  # time polynomial
            bf = np.sin(-np.pi*x[i-1]/2*self.K + np.pi/2)
        elif (self.nb_stocks < i < 2 * self.nb_stocks):
            k = i - self.nb_stocks - 1
            bf = np.exp(-x[k]/2)*(1-x[k])
        elif i == 2 *self.nb_stocks:  # time polynomial
            k = i - self.nb_stocks - 1
            bf = np.log(1 + self.T * (1-x[k]*self.K))
        elif (2 * self.nb_stocks < i < 3*self.nb_stocks):
            k = i - 2*self.nb_stocks -1
            bf = np.exp(-x[k]/2)*(1-2*x[k]+(x[k]**2)/2)
        elif i == 3*self.nb_stocks:
            k = i - 2*self.nb_stocks -1
            bf = (x[k]*self.K)**2
        return bf