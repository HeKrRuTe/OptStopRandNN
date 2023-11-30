""" American option price by Randomized Recurrent Least Square Monte Carlo.

It is the implementation of the Randomized Recurrent Least Square Monte Carlo
(RRLSM) introduced in (Optimal stopping via randomized neural networks,
Herrera, Krach, Ruyssen and Teichmann, 2021).
"""

import torch
import numpy as np
from optimal_stopping.algorithms.backward_induction import \
    backward_induction_pricer
from optimal_stopping.algorithms.utils import randomized_neural_networks
import sklearn.linear_model


class ReservoirRNNLeastSquarePricer(
    backward_induction_pricer.AmericanOptionPricer):
    def __init__(self, model, payoff, hidden_size=100, factors=(1.,1.,1.),
                 nb_epochs=None, nb_batches=None, train_ITM_only=True,
                 use_payoff_as_input=False):
        super().__init__(
            model, payoff, use_rnn=True, train_ITM_only=train_ITM_only,
            use_payoff_as_input=use_payoff_as_input)
        state_size = model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1
        self.RNN = randomized_neural_networks.randomRNN(
            state_size=state_size, hidden_size=hidden_size, factors=factors,
            extend=True)

    def compute_hs(self, stock_paths, var_paths=None):
        """
        Args:
         stock_paths: numpy array, shape [nb_paths, nb_stocks, nb_dates]
         var_paths: None or numpy array of shape [nb_paths, nb_stocks, nb_dates]

        Returns:
         hidden states (numpy array, shape [nb_dates, nb_paths, hidden_size])
        """
        if self.use_var:
            stock_paths = np.concatenate([stock_paths, var_paths], axis=1)
        x = torch.from_numpy(stock_paths).permute(2, 0, 1)
        x = x.type(torch.float32)
        hs = self.RNN(x).detach().numpy()
        return hs

    def calculate_continuation_value(self, values, immediate_exercise_value, h):
      """" See base class """
      if self.train_ITM_only:
          in_the_money = np.where(immediate_exercise_value[:self.split] > 0)
          in_the_money_all = np.where(immediate_exercise_value > 0)
      else:
          in_the_money = np.where(immediate_exercise_value[:self.split] < np.infty)
          in_the_money_all = np.where(immediate_exercise_value < np.infty)
      return_values = np.zeros(h.shape[0])
      reg_input = np.concatenate(
          [h, np.ones((len(h), 1))], axis=1)
      coefficients = np.linalg.lstsq(
          reg_input[in_the_money[0]], values[in_the_money[0]], rcond=None)
      return_values[in_the_money_all[0]] = np.dot(reg_input[in_the_money_all[0]],
                                                  coefficients[0])
      return return_values



class ReservoirRNNLeastSquarePricer2(ReservoirRNNLeastSquarePricer):
  def __init__(self, model, payoff, hidden_size=100, factors=(1.,1.,),
               nb_epochs=None, nb_batches=None, train_ITM_only=True,
               use_payoff_as_input=False):
      super().__init__(model, payoff, hidden_size, factors, nb_epochs,
                       nb_batches, train_ITM_only=train_ITM_only,
                       use_payoff_as_input=use_payoff_as_input)
      state_size = model.nb_stocks*(1+self.use_var)+self.use_payoff_as_input*1
      self.RNN = randomized_neural_networks.randomRNN(
          state_size=state_size, hidden_size=hidden_size, factors=factors,
          extend=False)


class ReservoirRNNLeastSquarePricer2Ridge(ReservoirRNNLeastSquarePricer2):
  def __init__(self, model, payoff, hidden_size=100, factors=(1.,1.,),
               ridge_coeff=1.,
               nb_epochs=None, nb_batches=None, train_ITM_only=True,
               use_payoff_as_input=False):
      super().__init__(model, payoff, hidden_size, factors, nb_epochs,
                       nb_batches, train_ITM_only=train_ITM_only,
                       use_payoff_as_input=use_payoff_as_input)
      self.alpha = ridge_coeff

  def calculate_continuation_value(self, values, immediate_exercise_value, h):
      """" See base class """
      if self.train_ITM_only:
          in_the_money = np.where(immediate_exercise_value[:self.split] > 0)
          in_the_money_all = np.where(immediate_exercise_value > 0)
      else:
          in_the_money = np.where(
              immediate_exercise_value[:self.split] < np.infty)
          in_the_money_all = np.where(immediate_exercise_value < np.infty)
      return_values = np.zeros(h.shape[0])
      reg_input = np.concatenate(
          [h, np.ones((len(h), 1))], axis=1)
      model = sklearn.linear_model.Ridge(alpha=self.alpha)
      model.fit(X=reg_input[in_the_money[0]], y=values[in_the_money[0]])
      continuation_values = model.predict(reg_input[in_the_money_all[0]])
      return continuation_values



