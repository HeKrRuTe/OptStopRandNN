""" Computes the American option price using least squares policy iteration.

It is the implementation of the least squares policy iteration (LSPI) studied in
(Learning Exercise Policies for American Options, Li , Szepesvari and Schuurmans, 2009).
"""

import numpy as np

from optimal_stopping.algorithms.reinforcement_learning import FQI


class LSPI(FQI.FQI):
  """ Computes the American option price using least squares policy iteration.
  """

  def get_indicator_stop(self, payoff, continuation_value):
    return payoff if payoff > continuation_value else 0

  def get_contribution_u(
      self, payoff, evaluated_bases, next_evaluated_bases, discount_factor,
      continuation_value):
    nb_base_fcts = len(evaluated_bases)
    indicator_continue = (next_evaluated_bases if payoff < continuation_value
                          else np.zeros(nb_base_fcts, dtype=float))
    return np.outer(evaluated_bases,
        evaluated_bases - discount_factor * indicator_continue)
