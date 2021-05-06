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


class ReservoirRNNLeastSquarePricer(
    backward_induction_pricer.AmericanOptionPricer):
    def __init__(self, model, payoff, hidden_size=100, factors=(1.,1.,1.),
                 nb_epochs=None, nb_batches=None, train_ITM_only=True):
        super().__init__(model, payoff, use_rnn=True, train_ITM_only=train_ITM_only)
        state_size = model.nb_stocks
        self.RNN = randomized_neural_networks.randomRNN(
            state_size=state_size, hidden_size=hidden_size, factors=factors,
            extend=False)

    def compute_hs(self, stock_paths):
        """
        Args:
         stock_paths (numpy array, shape [nb_paths, nb_stocks, nb_dates])

        Returns:
         hidden states (numpy array, shape [nb_dates, nb_paths, hidden_size])
        """
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
