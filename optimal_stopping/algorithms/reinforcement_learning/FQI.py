""" Computes the American option price using fitted Q-Iteration.

It is the implementation of the fitted Q-Iteration introduced in
(Regression methods for pricing complex American-style options:
A simple least-squares approach, Tsitsiklis and Van Roy, 2001).
"""

import math, time
import numpy as np

from optimal_stopping.algorithms.utils import basis_functions
from optimal_stopping.algorithms.reinforcement_learning import \
    reinforcement_learning_price


class FQI(reinforcement_learning_price.FQI_RL):
    """ Computes the american option price using fitted Q-Iteration (FQI)

    We recomand to use FQIFast which is a faster implemenation.
    """

    def __init__(self, model, payoff, nb_epochs=20, nb_batches=None):
        super().__init__(model, payoff, nb_epochs)
        self.bf = basis_functions.BasisFunctions(self.model.nb_stocks + 2)
        self.nb_base_fcts = self.bf.nb_base_fcts


    def evaluate_bases(self, stock_price, path, date, nb_dates):
        time = date/nb_dates
        stock_price_path_date = np.concatenate(
            [stock_price[path, :, date], np.array([time, 1 - time])]
        )
        return np.array([self.bf.base_fct(i, stock_price_path_date)
                         for i in range(self.bf.nb_base_fcts)])

    def evaluate_bases_all(self, stock_paths):
        raise NotImplementedError



class FQIFast(reinforcement_learning_price.FQI_RL):
    """ Computes the american option price using FQI.
    """

    def __init__(self, model, payoff, nb_epochs=20, nb_batches=None):
        super().__init__(model, payoff, nb_epochs)
        self.bf = basis_functions.BasisFunctions(self.model.nb_stocks + 2)
        self.nb_base_fcts = self.bf.nb_base_fcts

    def evaluate_bases(self, stock_paths, path, date, nb_dates):
        raise NotImplementedError

    def evaluate_bases_all(self, stock_price):
        """ see base class"""
        time = np.expand_dims(np.repeat(np.expand_dims(
            np.linspace(0, 1, stock_price.shape[2]), 0), stock_price.shape[0], axis=0), 1)
        stocks = np.concatenate([stock_price, time, 1-time], axis=1)
        stocks = np.transpose(stocks, (1, 0, 2))
        bf = np.concatenate(
            [np.expand_dims(self.bf.base_fct(i, stocks), axis=2)
             for i in range(self.bf.nb_base_fcts)], axis=2)
        return bf

    def price(self):
        t1 = time.time()
        stock_paths = self.model.generate_paths()
        self.split = int(len(stock_paths)/2)
        print("time path gen: {}".format(time.time() - t1), end=" ")
        nb_paths, nb_stocks, nb_dates = stock_paths.shape
        weights = np.zeros(self.nb_base_fcts, dtype=float)
        deltaT = self.model.maturity / nb_dates
        discount_factor = math.exp(-self.model.drift * deltaT)
        payoffs = self.payoff(stock_paths)
        eval_bases = self.evaluate_bases_all(stock_paths)

        for epoch in range(self.nb_epochs):
            continuation_value = np.dot(eval_bases[:self.split, 1:, :], weights)
            indicator_stop = np.maximum(payoffs[:self.split, 1:], continuation_value)
            matrixU = np.tensordot(eval_bases[:self.split, :-1, :],
                                   eval_bases[:self.split, :-1, :],
                                   axes=([0, 1],[0, 1]))
            vectorV = np.sum(
                eval_bases[:self.split, :-1, :] * discount_factor * np.repeat(
                    np.expand_dims(indicator_stop, axis=2), np.shape(eval_bases)[2],
                    axis=2),
                axis=(0, 1))
            weights = np.linalg.solve(matrixU, vectorV)

        continuation_value = np.maximum(np.dot(eval_bases, weights), 0)
        which = (payoffs > continuation_value)*1
        which[:, -1] = 1
        which[:, 0] = 0
        ex_dates = np.argmax(which, axis=1)
        _ex_dates = ex_dates + np.arange(len(ex_dates))*nb_dates
        prices = payoffs.reshape(-1)[_ex_dates] * discount_factor**ex_dates
        return max(np.mean(prices[self.split:]), payoffs[0, 0])


class FQIFastLaguerre(FQIFast):
    """ Computes the american option price using FQI
    with weighted Laguerre polynomials.
    """

    def __init__(self, model, payoff, nb_epochs=20, nb_batches=None):
        super().__init__(model, payoff, nb_epochs)
        self.bf = basis_functions.BasisFunctionsLaguerreTime(
            self.model.nb_stocks+1, T=model.maturity)
        self.nb_base_fcts = self.bf.nb_base_fcts

    def evaluate_bases_all(self, stock_price):
        """ see base class"""
        time = np.expand_dims(np.repeat(np.expand_dims(
            np.linspace(0, 1, stock_price.shape[2]), 0), stock_price.shape[0], axis=0), 1)
        stocks = np.concatenate([stock_price, time], axis=1)
        stocks = np.transpose(stocks, (1, 0, 2))
        bf = np.concatenate(
            [np.expand_dims(self.bf.base_fct(i, stocks), axis=2)
             for i in range(self.bf.nb_base_fcts)], axis=2)
        return bf
