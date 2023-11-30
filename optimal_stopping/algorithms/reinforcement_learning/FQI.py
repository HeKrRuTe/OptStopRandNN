""" Computes the American option price using fitted Q-Iteration.

It is the implementation of the fitted Q-Iteration introduced in
(Regression methods for pricing complex American-style options:
A simple least-squares approach, Tsitsiklis and Van Roy, 2001).
"""

import math, time
import numpy as np
import copy

from optimal_stopping.algorithms.utils import basis_functions
from optimal_stopping.algorithms.reinforcement_learning import \
    reinforcement_learning_price
from optimal_stopping.algorithms.utils import utilities
from optimal_stopping.run import configs

import sklearn.linear_model
from sklearn.linear_model import LinearRegression

class FQI(reinforcement_learning_price.FQI_RL):
    """Computes the american option price using fitted Q-Iteration (FQI)

    We recomand to use FQIFast which is a faster implemenation.
    """

    def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
                 use_payoff_as_input=False):
        super().__init__(model, payoff, nb_epochs,
                         use_payoff_as_input=use_payoff_as_input)
        self.bf = basis_functions.BasisFunctions(
            self.model.nb_stocks*(1+self.use_var) + 2 + self.use_payoff_as_input*1)
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

    def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
                 train_ITM_only=True, use_payoff_as_input=False):
        super().__init__(model, payoff, nb_epochs,
                         train_ITM_only=train_ITM_only,
                         use_payoff_as_input=use_payoff_as_input)
        self.bf = basis_functions.BasisFunctions(
            self.model.nb_stocks*(1+self.use_var) + 2 +
            self.use_payoff_as_input*1)
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

    def init_reg_weights(self, nb_base_fcts):
        self.weights = np.zeros(nb_base_fcts, dtype=float)

    def predict_reg(self, X):
        return np.dot(X, self.weights)

    def fit_reg(self, X, y):
        self.weights = np.linalg.solve(X, y)

    def price(self, train_eval_split=2):
        """
        see backward_induction_pricer.py
        """
        t1 = time.time()
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())
        stock_paths, var_paths = self.model.generate_paths()
        payoffs = self.payoff(stock_paths)
        stock_paths_with_payoff = np.concatenate(
            [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
        self.split = int(len(stock_paths)/train_eval_split)
        time_for_path_gen = time.time() - t1
        print("time path gen: {}".format(time.time() - t1), end=" ")
        nb_paths, nb_stocks, _ = stock_paths.shape
        nb_dates = self.model.nb_dates
        self.init_reg_weights(self.nb_base_fcts)
        deltaT = self.model.maturity / nb_dates
        discount_factor = math.exp(-self.model.rate * deltaT)
        if self.use_payoff_as_input:
            paths = stock_paths_with_payoff
        else:
            paths = stock_paths
        if self.use_var:
            paths = np.concatenate([paths, var_paths], axis=1)
        eval_bases = self.evaluate_bases_all(paths)

        for epoch in range(self.nb_epochs):
            continuation_value = self.predict_reg(eval_bases[:self.split, 1:, :])
            indicator_stop = np.maximum(payoffs[:self.split, 1:], continuation_value)
            matrixU = np.tensordot(eval_bases[:self.split, :-1, :],
                                   eval_bases[:self.split, :-1, :],
                                   axes=([0, 1],[0, 1]))
            vectorV = np.sum(
                eval_bases[:self.split, :-1, :] * discount_factor * np.repeat(
                    np.expand_dims(indicator_stop, axis=2), np.shape(eval_bases)[2],
                    axis=2),
                axis=(0, 1))
            self.fit_reg(matrixU, vectorV)
        if self.train_ITM_only:
            continuation_value = np.maximum(self.predict_reg(eval_bases), 0)
        else:
            continuation_value = self.predict_reg(eval_bases)
        which = (payoffs > continuation_value)*1
        which[:, -1] = 1
        which[:, 0] = 0
        ex_dates = np.argmax(which, axis=1)
        prices = np.take_along_axis(
            payoffs, np.expand_dims(ex_dates, axis=1), axis=1).reshape(-1) * \
                 discount_factor**ex_dates
        price = max(np.mean(prices[self.split:]), payoffs[0, 0])
        return price, time_for_path_gen

    def price_upper_lower_bound(self, verbose=0, train_eval_split=2):
        t1 = time.time()
        model = self.model
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())
        stock_paths, var_paths = self.model.generate_paths()
        payoffs = self.payoff(stock_paths)
        power = np.arange(0, model.nb_dates + 1)
        disc_factor = np.exp(
            (-model.rate) * model.maturity / model.nb_dates * power)
        disc_factors = np.repeat(
            np.expand_dims(disc_factor, axis=0), repeats=payoffs.shape[0],
            axis=0)
        payoffs = payoffs * disc_factors
        stock_paths_with_payoff = np.concatenate(
            [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
        self.split = int(len(stock_paths) / train_eval_split)
        time_for_path_gen = time.time() - t1
        print("time path gen: {}".format(time.time() - t1), end=" ")
        nb_paths, nb_stocks, _ = stock_paths.shape
        self.init_reg_weights(self.nb_base_fcts)
        if self.use_payoff_as_input:
            paths = stock_paths_with_payoff
        else:
            paths = stock_paths
        if self.use_var:
            paths = np.concatenate([paths, var_paths], axis=1)
        eval_bases = self.evaluate_bases_all(paths)

        for epoch in range(self.nb_epochs):
            continuation_value = self.predict_reg(
                eval_bases[:self.split, 1:, :])
            indicator_stop = np.maximum(payoffs[:self.split, 1:],
                                        continuation_value)
            matrixU = np.tensordot(eval_bases[:self.split, :-1, :],
                                   eval_bases[:self.split, :-1, :],
                                   axes=([0, 1], [0, 1]))
            vectorV = np.sum(
                eval_bases[:self.split, :-1, :] * np.repeat(
                    np.expand_dims(indicator_stop, axis=2),
                    np.shape(eval_bases)[2],
                    axis=2),
                axis=(0, 1))
            self.fit_reg(matrixU, vectorV)
        if self.train_ITM_only:
            continuation_value = np.maximum(self.predict_reg(eval_bases), 0)
        else:
            continuation_value = self.predict_reg(eval_bases)
        which = (payoffs > continuation_value) * 1
        which[:, -1] = 1
        which[:, 0] = 0
        ex_dates = np.argmax(which, axis=1)
        prices = np.take_along_axis(
            payoffs, np.expand_dims(ex_dates, axis=1), axis=1).reshape(-1)
        price = max(np.mean(prices[self.split:]), payoffs[0, 0])

        continuation_value[:, -1] = 0.
        M_diff = np.zeros_like(continuation_value)
        M_diff[:, 1:] = copy.deepcopy(np.maximum(
            payoffs[:, 1:], continuation_value[:, 1:])-continuation_value[:, :-1])
        M = np.cumsum(M_diff, axis=1)
        upper_bound = np.mean(
            np.max(payoffs[self.split:] - M[self.split:], axis=1))
        return price, upper_bound, time_for_path_gen

    def get_central_derivative(
            self, stock_paths, var_paths, discount_factor,
            stock_paths_p, var_paths_p, discount_factor_p,
            stock_paths_m, var_paths_m, discount_factor_m,
            eps, fd_freeze_exe_boundary=True, train_eval_split=2):
        """
        see backward_induction_pricer.py
        """
        payoffs = self.payoff(stock_paths)
        stock_paths_with_payoff = np.concatenate(
            [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
        payoffs_p = self.payoff(stock_paths_p)
        stock_paths_with_payoff_p = np.concatenate(
            [stock_paths_p, np.expand_dims(payoffs_p, axis=1)], axis=1)
        payoffs_m = self.payoff(stock_paths_m)
        stock_paths_with_payoff_m = np.concatenate(
            [stock_paths_m, np.expand_dims(payoffs_m, axis=1)], axis=1)
        self.split = int(len(stock_paths)/train_eval_split)
        nb_paths, nb_stocks, nb_dates = stock_paths.shape
        # weights = np.zeros(self.nb_base_fcts, dtype=float)
        self.init_reg_weights(self.nb_base_fcts)
        if self.use_payoff_as_input:
            paths = stock_paths_with_payoff
            paths_p = stock_paths_with_payoff_p
            paths_m = stock_paths_with_payoff_m
        else:
            paths = stock_paths
            paths_p = stock_paths_p
            paths_m = stock_paths_m
        if self.use_var:
            paths = np.concatenate([paths, var_paths], axis=1)
            paths_p = np.concatenate([paths_p, var_paths_p], axis=1)
            paths_m = np.concatenate([paths_m, var_paths_m], axis=1)
        eval_bases = self.evaluate_bases_all(paths)
        eval_bases_p = self.evaluate_bases_all(paths_p)
        eval_bases_m = self.evaluate_bases_all(paths_m)

        for epoch in range(self.nb_epochs):
            continuation_value = self.predict_reg(eval_bases[:self.split, 1:, :])
            indicator_stop = np.maximum(payoffs[:self.split, 1:], continuation_value)
            matrixU = np.tensordot(eval_bases[:self.split, :-1, :],
                                   eval_bases[:self.split, :-1, :],
                                   axes=([0, 1],[0, 1]))
            vectorV = np.sum(
                eval_bases[:self.split, :-1, :] * discount_factor * np.repeat(
                    np.expand_dims(indicator_stop, axis=2), np.shape(eval_bases)[2],
                    axis=2),
                axis=(0, 1))
            self.fit_reg(matrixU, vectorV)
        continuation_value = np.maximum(self.predict_reg(eval_bases), 0)

        if not fd_freeze_exe_boundary:
            for epoch in range(self.nb_epochs):
                continuation_value_p = self.predict_reg(
                    eval_bases_p[:self.split, 1:, :])
                indicator_stop = np.maximum(
                    payoffs_p[:self.split, 1:], continuation_value_p)
                matrixU = np.tensordot(eval_bases_p[:self.split, :-1, :],
                                       eval_bases_p[:self.split, :-1, :],
                                       axes=([0, 1],[0, 1]))
                vectorV = np.sum(
                    eval_bases_p[:self.split, :-1, :] * discount_factor_p *
                    np.repeat(np.expand_dims(indicator_stop, axis=2),
                              np.shape(eval_bases_p)[2], axis=2),
                    axis=(0, 1))
                self.fit_reg(matrixU, vectorV)
            continuation_value_p = np.maximum(self.predict_reg(eval_bases_p), 0)

            for epoch in range(self.nb_epochs):
                continuation_value_m = self.predict_reg(
                    eval_bases_m[:self.split, 1:, :])
                indicator_stop = np.maximum(
                    payoffs_m[:self.split, 1:], continuation_value_m)
                matrixU = np.tensordot(eval_bases_m[:self.split, :-1, :],
                                       eval_bases_m[:self.split, :-1, :],
                                       axes=([0, 1],[0, 1]))
                vectorV = np.sum(
                    eval_bases_m[:self.split, :-1, :] * discount_factor_m *
                    np.repeat(np.expand_dims(indicator_stop, axis=2),
                              np.shape(eval_bases_m)[2], axis=2),
                    axis=(0, 1))
                self.fit_reg(matrixU, vectorV)
            continuation_value_m = np.maximum(self.predict_reg(eval_bases_m), 0)
        else:
            continuation_value_p = continuation_value
            continuation_value_m = continuation_value

        which = (payoffs > continuation_value)*1
        which[:, -1] = 1
        which[:, 0] = 0
        which_p = (payoffs_p > continuation_value_p)*1
        which_p[:, -1] = 1
        which_p[:, 0] = 0
        which_m = (payoffs_m > continuation_value_m)*1
        which_m[:, -1] = 1
        which_m[:, 0] = 0
        ex_dates = np.argmax(which, axis=1)
        _ex_dates = ex_dates + np.arange(len(ex_dates))*nb_dates
        ex_dates_p = np.argmax(which_p, axis=1)
        _ex_dates_p = ex_dates_p + np.arange(len(ex_dates_p))*nb_dates
        ex_dates_m = np.argmax(which_m, axis=1)
        _ex_dates_m = ex_dates_m + np.arange(len(ex_dates_m))*nb_dates
        prices = payoffs.reshape(-1)[_ex_dates] * discount_factor**ex_dates
        prices_p = payoffs_p.reshape(-1)[_ex_dates_p] \
                   * discount_factor_p**ex_dates_p
        prices_m = payoffs_m.reshape(-1)[_ex_dates_m] \
                   * discount_factor_m**ex_dates_m

        price = [np.mean(prices[self.split:]), np.mean(prices_p[self.split:]),
                 np.mean(prices_m[self.split:])]
        derivative = np.mean(
            (prices_p[self.split:] - prices_m[self.split:])/(2*eps))

        return price, derivative

    def get_regression(self, spot, eps, d, dW):
        """
        see backward_induction_pricer.py
        """
        t1 = time.time()
        self.model.spot = spot
        X0 = np.random.normal(loc=spot, scale=eps,
                              size=(self.model.nb_paths,1))
        X0 = np.repeat(X0, repeats=self.model.nb_stocks, axis=1)
        stock_paths, var_paths = self.model.generate_paths(dW=dW, X0=X0)
        time_path_gen = time.time() - t1

        payoffs = self.payoff(stock_paths)
        stock_paths_with_payoff = np.concatenate(
            [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
        self.split = len(stock_paths)
        nb_paths, nb_stocks, nb_dates = stock_paths.shape
        self.init_reg_weights(self.nb_base_fcts)
        deltaT = self.model.maturity / nb_dates
        discount_factor = math.exp(-self.model.rate * deltaT)
        if self.use_payoff_as_input:
            paths = stock_paths_with_payoff
        else:
            paths = stock_paths
        if self.use_var:
            paths = np.concatenate([paths, var_paths], axis=1)
        eval_bases = self.evaluate_bases_all(paths)

        for epoch in range(self.nb_epochs):
            continuation_value = self.predict_reg(eval_bases[:self.split, 1:, :])
            indicator_stop = np.maximum(payoffs[:self.split, 1:], continuation_value)
            matrixU = np.tensordot(eval_bases[:self.split, :-1, :],
                                   eval_bases[:self.split, :-1, :],
                                   axes=([0, 1],[0, 1]))
            vectorV = np.sum(
                eval_bases[:self.split, :-1, :] * discount_factor * np.repeat(
                    np.expand_dims(indicator_stop, axis=2), np.shape(eval_bases)[2],
                    axis=2),
                axis=(0, 1))
            self.fit_reg(matrixU, vectorV)
        continuation_value = np.maximum(self.predict_reg(eval_bases), 0)
        which = (payoffs > continuation_value)*1
        which[:, -1] = 1
        which[:, 0] = 0
        ex_dates = np.argmax(which, axis=1)
        _ex_dates = ex_dates + np.arange(len(ex_dates))*nb_dates

        prices = payoffs.reshape(-1)[_ex_dates] * discount_factor**ex_dates

        # fit regression to values
        b, b_d, b_g = utilities.get_poly_basis_and_derivatives(X=X0[:, :1], d=d)
        b_val, b_d_val, b_g_val = utilities.get_poly_basis_and_derivatives(
            X=np.array([[spot]]), d=d)
        linreg = LinearRegression(fit_intercept=False)
        res = linreg.fit(X=b, y=prices)

        price = linreg.predict(X=b_val)[0]
        delta = linreg.predict(X=b_d_val)[0]
        gamma = linreg.predict(X=b_g_val)[0]

        return price, delta, gamma, time_path_gen



class FQIFastDeg1(FQIFast):

    def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
                 train_ITM_only=True,
                 use_payoff_as_input=False):
        super().__init__(model, payoff, nb_epochs,
                         train_ITM_only=train_ITM_only,
                         use_payoff_as_input=use_payoff_as_input)
        self.bf = basis_functions.BasisFunctionsDeg1(
            self.model.nb_stocks*(1+self.use_var)+2+self.use_payoff_as_input*1)
        self.nb_base_fcts = self.bf.nb_base_fcts


class FQIFastLaguerre(FQIFast):
    """ Computes the american option price using FQI
    with weighted Laguerre polynomials.
    """

    def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
                 train_ITM_only=True,
                 use_payoff_as_input=False):
        super().__init__(model, payoff, nb_epochs,
                         train_ITM_only=train_ITM_only,
                         use_payoff_as_input=use_payoff_as_input)
        self.bf = basis_functions.BasisFunctionsLaguerreTime(
            self.model.nb_stocks*(1+self.use_var)+1+self.use_payoff_as_input*1,
            T=model.maturity)
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


class FQIFastRidge(FQIFast):
    """ Computes the american option price using FQI.
    """

    def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
                 ridge_coeff=1., train_ITM_only=True,
                 use_payoff_as_input=False):
        super().__init__(model=model, payoff=payoff, nb_epochs=nb_epochs,
                         nb_batches=nb_batches,
                         train_ITM_only=train_ITM_only,
                         use_payoff_as_input=use_payoff_as_input)
        self.init_reg_model(ridge_coeff=ridge_coeff)
        # print("use ridge coeff: {}".format(ridge_coeff))

    def init_reg_model(self, ridge_coeff):
        self.reg_model = sklearn.linear_model.Ridge(
            alpha=ridge_coeff, fit_intercept=False)
        print("ridge")

    def init_reg_weights(self, nb_base_fcts):
        self.reg_model.fit(X=np.eye(nb_base_fcts), y=np.zeros(nb_base_fcts))

    def predict_reg(self, X):
        # if len(X.shape) == 3:
        #     return np.array([self.reg_model.predict(X[i]) for i in range(X.shape[0])])
        # elif len(X.shape) == 2:
        #     return self.reg_model.predict(X)
        # else:
        #     raise ValueError

        shape = X.shape
        assert len(shape) > 1
        X = X.reshape(-1, shape[-1])
        pred = self.reg_model.predict(X)
        return pred.reshape(shape[:-1])

    def fit_reg(self, X, y):
        self.reg_model.fit(X=X, y=y)


class FQIFastLasso(FQIFastRidge):
    """ Computes the american option price using FQI.
    """

    def init_reg_model(self, ridge_coeff):
        self.reg_model = sklearn.linear_model.Lasso(
            alpha=ridge_coeff, fit_intercept=False)
        print("lasso")
