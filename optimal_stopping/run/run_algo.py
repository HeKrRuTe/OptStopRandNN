# Lint as: python3
"""
Main module to run the algorithms.
"""
import os

import atexit
import csv
import itertools
import multiprocessing
import socket
import random
import time
import psutil

# absl needs to be upgraded to >= 0.10.0, otherwise joblib might not work
from absl import app
from absl import flags
import numpy as np
import shutil

from optimal_stopping.utilities import configs_getter
from optimal_stopping.algorithms.backward_induction import DOS
from optimal_stopping.payoffs import payoff
from optimal_stopping.algorithms.backward_induction import LSM
from optimal_stopping.algorithms.backward_induction import RLSM
from optimal_stopping.algorithms.backward_induction import RRLSM
from optimal_stopping.data import stock_model
from optimal_stopping.algorithms.backward_induction import NLSM
from optimal_stopping.algorithms.reinforcement_learning import RFQI
from optimal_stopping.algorithms.reinforcement_learning import FQI
from optimal_stopping.algorithms.reinforcement_learning import LSPI
from optimal_stopping.run import write_figures


import joblib

# GLOBAL CLASSES
class SendBotMessage:
    def __init__(self):
        pass

    @staticmethod
    def send_notification(text, *args, **kwargs):
        print(text)

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    SBM = SendBotMessage()

NUM_PROCESSORS = multiprocessing.cpu_count()
if 'ada-' in socket.gethostname() or 'arago' in socket.gethostname():
    SERVER = True
    NB_JOBS = int(NUM_PROCESSORS) - 1
else:
    SERVER = False
    NB_JOBS = int(NUM_PROCESSORS) - 1


SEND = False
if SERVER:
    SEND = True





FLAGS = flags.FLAGS

flags.DEFINE_list("nb_stocks", None, "List of number of Stocks")
flags.DEFINE_list("algos", None, "Name of the algos to run.")
flags.DEFINE_bool("print_errors", False, "Set to True to print errors if any.")
flags.DEFINE_integer("nb_jobs", NB_JOBS, "Number of CPUs to use parallelly")
flags.DEFINE_bool("generate_pdf", False, "Whether to generate latex tables")

_CSV_HEADERS = ['algo', 'model', 'payoff', 'drift', 'volatility', 'mean',
                'speed', 'correlation', 'hurst', 'nb_stocks',
                'nb_paths', 'nb_dates', 'spot', 'strike', 'dividend',
                'maturity', 'nb_epochs', 'hidden_size', 'factors',
                'ridge_coeff',
                'train_ITM_only', 'use_path',
                'price', 'duration']

_PAYOFFS = {
    "MaxPut": payoff.MaxPut,
    "MaxCall": payoff.MaxCall,
    "GeometricPut": payoff.GeometricPut,
    "BasketCall": payoff.BasketCall,
    "Identity": payoff.Identity,
    "Max": payoff.Max,
    "Mean": payoff.Mean,
}

_STOCK_MODELS = {
    "BlackScholes": stock_model.BlackScholes,
    "FractionalBlackScholes": stock_model.FractionalBlackScholes,
    "FractionalBrownianMotion": stock_model.FractionalBrownianMotion,
    'FractionalBrownianMotionPathDep':
        stock_model.FractionalBrownianMotionPathDep,
    "Heston": stock_model.Heston,
}

_ALGOS = {
    "LSM": LSM.LeastSquaresPricer,
    "LSMLaguerre": LSM.LeastSquarePricerLaguerre,
    "LSMRidge": LSM.LeastSquarePricerRidge,
    "FQI": FQI.FQIFast,
    "FQILaguerre": FQI.FQIFastLaguerre,
    "LSPI": LSPI.LSPI,  # TODO: this is a slow version -> update similar to FQI

    "NLSM": NLSM.NeuralNetworkPricer,
    "DOS": DOS.DeepOptimalStopping,

    "RLSM": RLSM.ReservoirLeastSquarePricerFast,
    "RLSMRidge": RLSM.ReservoirLeastSquarePricerFastRidge,

    "RRLSM": RRLSM.ReservoirRNNLeastSquarePricer,

    "RFQI": RFQI.FQI_ReservoirFast,
    "RRFQI": RFQI.FQI_ReservoirFastRNN,
}

_NUM_FACTORS = {
    "RRLSMmix": 3,
    "RRLSM": 2,
    "RLSM": 1,
}



def init_seed():
  random.seed(0)
  np.random.seed(0)


def _run_algos():
  fpath = os.path.join(os.path.dirname(__file__), "../../output/metrics_draft",
                       f'{int(time.time()*1000)}.csv')
  tmp_dirpath = f'{fpath}.tmp_results'
  os.makedirs(tmp_dirpath, exist_ok=True)
  atexit.register(shutil.rmtree, tmp_dirpath)
  tmp_files_idx = 0

  delayed_jobs = []

  nb_stocks_flag = [int(nb) for nb in FLAGS.nb_stocks or []]
  for config_name, config in configs_getter.get_configs():
    print(f'Config {config_name}', config)
    config.algos = [a for a in config.algos
                    if FLAGS.algos is None or a in FLAGS.algos]
    if nb_stocks_flag:
      config.nb_stocks = [a for a in config.nb_stocks
                          if a in nb_stocks_flag]
    combinations = list(itertools.product(
        config.algos, config.dividends, config.maturities, config.nb_dates,
        config.nb_paths, config.nb_stocks, config.payoffs, config.drift,
        config.spots, config.stock_models, config.strikes, config.volatilities,
        config.mean, config.speed, config.correlation, config.hurst,
        config.nb_epochs, config.hidden_size, config.factors,
        config.ridge_coeff,
        config.train_ITM_only, config.use_path))
    # random.shuffle(combinations)
    for params in combinations:
      for i in range(config.nb_runs):
        tmp_file_path = os.path.join(tmp_dirpath, str(tmp_files_idx))
        tmp_files_idx += 1
        delayed_jobs.append(joblib.delayed(_run_algo)(
            tmp_file_path, *params, fail_on_error=FLAGS.print_errors)
        )

  print(f"Running {len(delayed_jobs)} tasks using "
        f"{FLAGS.nb_jobs}/{NUM_PROCESSORS} CPUs...")
  joblib.Parallel(n_jobs=FLAGS.nb_jobs)(delayed_jobs)

  print(f'Writing results to {fpath}...')
  with open(fpath, "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=_CSV_HEADERS)
    writer.writeheader()
    for idx in range(tmp_files_idx):
      tmp_file_path = os.path.join(tmp_dirpath, str(idx))
      try:
        with open(tmp_file_path,  "r") as read_f:
          csvfile.write(read_f.read())
      except FileNotFoundError:
        pass

  return fpath


def _run_algo(
        metrics_fpath, algo, dividend, maturity, nb_dates, nb_paths,
        nb_stocks, payoff, drift, spot, stock_model, strike, volatility, mean,
        speed, correlation, hurst, nb_epochs, hidden_size=10,
        factors=(1.,1.,1.), ridge_coeff=1.,
        train_ITM_only=True, use_path=False,
        fail_on_error=False):
  """
  This functions runs one algo for option pricing. It is called by _run_algos()
  which is called in main(). Below the inputs are listed which have to be
  specified in the config that is passed to main().

  Args:
   metrics_fpath: file path, automatically generated & passed by
            _run_algos()
   algo (str): the algo to train. See dict _ALGOS above.
   dividend (float): the dividend of the stock model.
   maturity (float): the maturity of the option.
   nb_dates (int): number of equidistance dates at which option can be
            exercised up to maturity.
   nb_paths (int): number of paths that are simulated from stock model.
            Half is used to learn the weigths, half to estimate the option
            price.
   nb_stocks (int): number of stocks used for the option (e.g. max call).
   payoff (str): see dict _PAYOFFS.
   drift (float): the drift of the stock model
   spot (float): the value of the stocks at t=0.
   stock_model (str): see dict _STOCK_MODELS.
   strike (float): the strike price of the option, if used by the payoff.
   volatility (float): the volatility of the stock model.
   mean (float): parameter for Heston stock model.
   speed (float): parameter for Heston stock model.
   correlation (float): parameter for Heston stock model.
   hurst (float): in (0,1) the hurst parameter for the
            FractionalBrownianMotion and FractionalBrownianMotionPathDep stock
            models.
   nb_epochs (int): number of epochs to train the algos which are based on
            iterative updates (FQI, RFQI etc.) or SGD (NLSM, DOS). Otherwise
            unused.
   hidden_size (int): the number of nodes in the hidden layers of NN based
            algos.
   factors (list of floats, optional. Contains scaling coeffs for the
            randomized NN (i.e. scaling of the randomly sampled and fixed
            weights -> changes the std of the sampling distribution). Depending
            on the algo, the factors are used differently. See there directly.
   ridge_coeff (float, regression coeff for the algos using Ridge
            regression.
   train_ITM_only (bool): whether to train weights on all paths or only on
            those where payoff is positive (i.e. where it makes sense to stop).
            This should be set to False when using FractionalBrownianMotion with
            Identity payoff, since there the payoff can actually become negative
            and therefore training on those paths is important.
   use_path (bool): for DOS algo only. If true, the algo uses the entire
            path up to the current time of the stock (instead of the current
            value only) as input. This is used for Non-Markovian stock models,
            i.e. FractionalBrownianMotion, where the decisions depend on the
            history.
   fail_on_error (bool): whether to continue when errors occure or not.
            Automatically passed from _run_algos(), with the value of
            FLAGS.print_errors.
  """
  print(algo, spot, volatility, maturity, nb_paths, '... ', end="")
  payoff_ = _PAYOFFS[payoff](strike)
  stock_model_ = _STOCK_MODELS[stock_model](
      drift=drift, volatility=volatility, mean=mean, speed=speed, hurst=hurst,
      correlation=correlation, nb_stocks=nb_stocks,
      nb_paths=nb_paths, nb_dates=nb_dates,
      spot=spot, dividend=dividend,
      maturity=maturity)
  if algo in ['NLSM']:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            hidden_size=hidden_size,
                            train_ITM_only=train_ITM_only)
  elif algo in ["DOS"]:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            hidden_size=hidden_size, use_path=use_path)
  elif algo in ["RLSM", "RRLSM", "RRFQI", "RFQI",]:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            hidden_size=hidden_size, factors=factors,
                            train_ITM_only=train_ITM_only)
  elif algo in ["RLSMRidge"]:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            hidden_size=hidden_size, factors=factors,
                            train_ITM_only=train_ITM_only,
                            ridge_coeff=ridge_coeff)
  elif algo in ["LSM", "LSMLaguerre"]:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            train_ITM_only=train_ITM_only)
  elif algo in ["LSMRidge"]:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            train_ITM_only=train_ITM_only,
                            ridge_coeff=ridge_coeff)
  else:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs)

  t_begin = time.time()
  try:
    price = pricer.price()
    duration = time.time() - t_begin
  except BaseException as err:
    if fail_on_error:
      raise
    print(err)
    return
  metrics_ = {}
  metrics_['algo'] = algo
  metrics_['model'] = stock_model
  metrics_['payoff'] = payoff
  metrics_['drift'] = drift
  metrics_['volatility'] = volatility
  metrics_['mean'] = mean
  metrics_['speed'] = speed
  metrics_['correlation'] = correlation
  metrics_['hurst'] = hurst
  metrics_['nb_stocks'] = nb_stocks
  metrics_['nb_paths'] = nb_paths
  metrics_['nb_dates'] = nb_dates
  metrics_['spot'] = spot
  metrics_['strike'] = strike
  metrics_['dividend'] = dividend
  metrics_['maturity'] = maturity
  metrics_['price'] = price
  metrics_['duration'] = duration
  metrics_['hidden_size'] = hidden_size
  metrics_['factors'] = factors
  metrics_['ridge_coeff'] = ridge_coeff
  metrics_['nb_epochs'] = nb_epochs
  metrics_['train_ITM_only'] = train_ITM_only
  metrics_['use_path'] = use_path
  print("price: ", price, "duration: ", duration)
  with open(metrics_fpath, "w") as metrics_f:
    writer = csv.DictWriter(metrics_f, fieldnames=_CSV_HEADERS)
    writer.writerow(metrics_)



def main(argv):
  del argv

  try:
      if SEND:
          SBM.send_notification(
              text='start running AMC2 with config:\n{}'.format(FLAGS.configs),
              chat_id="-399803347"
          )

      filepath = _run_algos()

      if FLAGS.generate_pdf:
          write_figures.write_figures()
          write_figures.generate_pdf()

      if SEND:
          time.sleep(1)
          SBM.send_notification(
              text='finished',
              files=[filepath],
              chat_id="-399803347"
          )
  except Exception as e:
      if SEND:
          SBM.send_notification(
              text='ERROR\n{}'.format(e),
              chat_id="-399803347"
          )
      else:
          print('ERROR\n{}'.format(e))




if __name__ == "__main__":
  app.run(main)
