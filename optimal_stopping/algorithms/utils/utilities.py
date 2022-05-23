"""
author: Florian Krach
"""

import numpy as np


def get_poly_basis_and_derivatives(X, d):
    """
    computes the polynomial regression basis of the input X of degree d
    together with its first and second derivative
    """
    nb_stock = X.shape[1]
    poly_basis = np.ones((X.shape[0], (1+d)*nb_stock))
    poly_basis_delta = np.zeros((X.shape[0], (1+d)*nb_stock))
    poly_basis_gamma = np.zeros((X.shape[0], (1+d)*nb_stock))
    for j in range(nb_stock):
        for i in range(1, d+1):
            poly_basis[:, i+j*nb_stock] = X[:,j]**i
            poly_basis_delta[:, i+j*nb_stock] = float(i)*X[:,j]**(i-1)
            if i>1:
                poly_basis_gamma[:, i+j*nb_stock] = float(i*(i-1))*X[:,j]**(i-2)
            else:
                poly_basis_gamma[:, i+j*nb_stock] = 0.
    return poly_basis, poly_basis_delta, poly_basis_gamma


def compute_gamma_via_BS_PDE(
        price, delta, theta, rate, vola, spot, dividend=0.):
    """
    use the Black-Scholes PDE
    (https://en.wikipedia.org/wiki/Black–Scholes_equation) to compute the value
    of gamma out of the price, delta and theta.
    For the extension with dividend see equation (2) in:
    https://www.math.tamu.edu/~mike.stecher/425/Sp12/optionsForDividendStocks.pdf
    """
    return (rate*price - theta - (rate-dividend)*spot*delta)*2 / \
           (vola**2 * spot**2)