"""Write PDF table to compare algorithms results."""
import copy
import os.path
from typing import Iterable

import pandas as pd
import numpy as np
import tensorflow as tf

from optimal_stopping.run import configs
from optimal_stopping.utilities import read_data

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_list("rm_from_index", None, "List of keys to remove from index")


_PDF_TABLE_TMPL = r"""
\begin{table*}[!h]
\center
\scalebox{0.60}{
%(table)s
}
\caption{%(caption)s
}
\label{%(label)s}
\end{table*}
"""

ALGOS_ORDER = [
  "LN2", "LNfast", "LSPI",
  "LSM", "LSMRidge",  "LSMLaguerre", "LSMDeg1",
  "DOS", "pathDOS", "NLSM",
  "RLSM", "RLSMTanh", "RLSMRidge",
  "RLSMElu", "RLSMSilu", "RLSMGelu", "RLSMSoftplus",
  "RRLSM", "RRLSMmix",
  "FQI", "FQIR", "FQIRidge", "FQILasso", "FQILaguerre", "FQIDeg1",
  "RFQI", "RFQITanh", "RFQIRidge", "RFQILasso", "RRFQI", 'pathRFQI',
  "RFQISoftplus",
  "EOP", "B", "Trinomial",
]
COLUMNS_ORDER = ["price", "duration"]

USE_PAYOFF_FOR_ALGO = {
  "LSM": True,
  "LSMRidge": True,
  "LSMLaguerre": True,
  "LSMDeg1": True,
  "DOS": True,
  "pathDOS": True,
  "NLSM": True,
  "RLSM": True,
  "RLSMTanh": True,
  "RLSMRidge": True,
  "RLSMElu": False,
  "RLSMSilu": False,
  "RLSMGelu": False,
  "RLSMSoftplus": False,
  "RRLSM": True,
  "RRLSMmix": True,
  "FQI": False,
  "FQIR": False,
  "FQIRidge": False,
  "FQILasso": False,
  "FQILaguerre": False,
  "FQIDeg1": False,
  "RFQI": False,
  "RFQITanh": False,
  "RFQIRidge": False,
  "RFQILasso": False,
  "RFQISoftplus": False,
  "RRFQI": False,
  'pathRFQI': False,
  "EOP": False,
  "B": False,
  "Trinomial": False,
}


def write_table_price(label: str, config: configs._DefaultConfig, get_df=False):
  df = _write_table_for(
    label, config, ["price"], "Prices returned by each algo", get_df=get_df)
  if get_df:
    return df


def write_table_duration(label: str, config: configs._DefaultConfig):
  _write_table_for(label, config, ["duration"], "Duration (s) of each algo")


def write_table_price_duration(label: str, config: configs._DefaultConfig):
  _write_table_for(label, config, ["price", "duration"],
      "Algorithms were run "
      "$10$ times and the mean and the standard deviation (in parenthesis) of "
      "the prices as well as the median of the computation time are given.")


def _human_time_delta(delta):
  hours = delta // 3600
  minutes = (delta - hours*3600) // 60
  seconds = int(delta - hours*3600 - minutes*60)
  return "".join([
    "%dh" %hours if hours else "  ",
    "%2dm" % minutes if minutes else "  ",
    "%2ds" % seconds
  ])


def _write_table_for(
        label: str, config: configs._DefaultConfig,
        column_names: Iterable[str], caption: str,
        get_df=False, which_time="comp_time",
        get_max_usepayoff=False, get_algo_specific_usepayoff=True,
):
  df = read_data.read_csvs(config, remove_duplicates=False)
  if which_time != 'duration' and 'duration' in column_names:
    df.drop(columns=['duration'], inplace=True)
    df.rename(columns={which_time: 'duration'}, inplace=True)
  df = df.filter(items=column_names)

  # replace NaNs by "no_val", such that grouping etc still works
  df.reset_index(inplace=True)
  df[read_data.INDEX] = df[read_data.INDEX].replace(np.nan, "no_val")
  rmfi = FLAGS.rm_from_index
  index = read_data.INDEX
  if rmfi is not None:
    df.drop(columns=rmfi, inplace=True)
    for i in rmfi:
      index.remove(i)
  all_algos = np.unique(df["algo"].values)
  df.set_index(index, inplace=True)

  if 'price' in column_names:
    mean_price = df.groupby(df.index)['price'].mean()
    std = df.groupby(df.index)['price'].std()
    #size = df.groupby(df.index)['price'].size()[0]
    #conf_interval_price = 1.96 * std  / math.sqrt(size)

  if 'duration' in column_names:
    median_duration = df.groupby(df.index)['duration'].median()

  df = df[~df.index.duplicated(keep='last')]

  if 'duration' in column_names:
    try:
      df['duration'] = median_duration
      df['duration'] = [_human_time_delta(sec)
                        for sec in df['duration']]
    except Exception:
      df['duration'] = None

  # print(df)
  if 'price' in column_names:
    if get_df:
      df['price'] = mean_price
    else:
      df['mean_price'] = mean_price
      df['std_price'] = std #conf_interval_price
      df['price'] = ['%.2f (%.2f)' % ms
                     for ms in zip(df['mean_price'], df['std_price'])]
      df = df.drop('std_price', 'columns')

      # for each algo keep the maximum value of using the payoff as input or not
      if get_max_usepayoff and len(config.use_payoff_as_input) == 2:
        ii = np.where(np.array(index) == "use_payoff_as_input")[0][0]
        for ind in df.index:
          ind1 = list(ind)
          ind2 = copy.copy(ind1)
          ind2[ii] = not ind2[ii]
          try:
            if df.loc[tuple(ind1), "price"] > df.loc[tuple(ind2), "price"]:
              df.drop(index=tuple(ind2), inplace=True)
            else:
              df.drop(index=tuple(ind1), inplace=True)
          except KeyError as e:
            pass
        df.reset_index(inplace=True)
        index.remove("use_payoff_as_input")
        df.drop(columns="use_payoff_as_input", inplace=True)
        df.set_index(index, inplace=True)
      # for each algo keep the algo-specific value of using the payoff or not
      elif get_algo_specific_usepayoff and len(config.use_payoff_as_input) == 2:
        ii = np.where(np.array(index) == "use_payoff_as_input")[0][0]
        jj = np.where(np.array(index) == "algo")[0][0]
        for ind in df.index:
          ind1 = list(ind)
          ind2 = copy.copy(ind1)
          ind2[ii] = not ind2[ii]
          try:
            if ind1[ii] == USE_PAYOFF_FOR_ALGO[ind1[jj]]:
              df.drop(index=tuple(ind2), inplace=True)
            elif tuple(ind2) in df.index:
              df.drop(index=tuple(ind1), inplace=True)
          except KeyError as e:
            pass
        df.reset_index(inplace=True)
        index.remove("use_payoff_as_input")
        df.drop(columns="use_payoff_as_input", inplace=True)
        df.set_index(index, inplace=True)

      df2 = df["mean_price"]
      df = df.drop('mean_price', 'columns')

  df, global_params_caption = read_data.extract_single_value_indexes(df)
  df = df.unstack('algo')

  def my_key(index):
    if index.name == 'algo':
      return pd.Index([ALGOS_ORDER.index(algo) for algo in index], name='algo')
    return pd.Index([COLUMNS_ORDER.index(name) for name in index], name='')
  try:
    df = df.sort_index(key=my_key, axis='columns')
  except Exception:
    df = df.sort_index(key=my_key)
    df = df.to_frame().T
  print(df)

  # get relative errors if reference price exists
  df2, global_params_caption = read_data.extract_single_value_indexes(df2)
  df2 = df2.unstack('algo')
  df2 = df2["mean_price"]
  if "EOP" in all_algos:
    ref_algo = "EOP"
    df2[ref_algo].fillna(method="ffill", inplace=True)
  elif "B" in all_algos:
    ref_algo = "B"
  else:
    ref_algo = None
  for a,b in [["LSM", "RLSM"], ["FQI", "RFQI"], ["DOS", "RFQI"], ["FQI", "DOS"],
              ["LSM", "RFQI"], ["DOS", "RLSM"], ["NLSM", "RLSM"],
              ["RLSM", "RRLSM"], ["pathDOS", "RLSM"],]:
    try:
      print(a,b)
      print((df2[a] - df2[b])/df2[a])
    except Exception:
      pass
  if ref_algo:
    for a in all_algos:
      if a != ref_algo:
        df2[a] = np.abs(df2[ref_algo] - df2[a])/df2[ref_algo]
    print(df2)
    df2.reset_index(inplace=True)
    print(df2.loc[df2["nb_stocks"] <=100].max(axis=0))
    print(df2.loc[df2["nb_stocks"] > 100].max(axis=0))



  if get_df:
    return df

  algos = df.columns.get_level_values("algo").unique()

  # bold_algos = ["RLSM", "RRLSM", "RFQI"]
  bold_algos = []
  _table_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), f"../../../latex/tables_draft/"))
  if not os.path.exists(_table_path):
    os.makedirs(_table_path)
  table_path = os.path.abspath(os.path.join(
      os.path.dirname(__file__), f"../../../latex/tables_draft/{label}.tex"))
  global_params_caption = global_params_caption.replace('_', '\\_')
  caption = (f"{caption}. {global_params_caption}.")
  mcol_format = ' '.join('>{\\bfseries}r' if algo in bold_algos else 'r'
                         for algo in algos)
  ind = df.index.names
  if ind == [None]:
    col_format = ('|' +
                  '|'.join([mcol_format] * len(column_names)) +
                  '|')
  else:
    col_format  =  ('|' + 'c' * len(ind) + '|' +
                    '|'.join([mcol_format] * len(column_names)) +
                    '|')
  pdf_table = _PDF_TABLE_TMPL % {
      "table": df.to_latex(
          na_rep="-", multirow=True, multicolumn=True,
          multicolumn_format='c |',
          float_format="%.2f",
          column_format=col_format),
      "caption": caption,
      "label": label,
  }

  try:
    new_header = ' & '.join(df.index.names + list(algos)*2) + '\\\\'
    oneline = False
  except Exception:
    new_header = ' & '.join(list(algos) * 2) + '\\\\'
    oneline = True
  new_lines = []
  for line in pdf_table.split('\n'):
    if 'algo &' in line:
      new_lines.append(new_header)
    elif line.startswith('nb\\_stocks &') or line.startswith('hidden\\_size &') \
        or line.startswith('maturity &') or line.startswith('payoff ') \
        or line.startswith('model '):
      continue
    elif oneline and line.startswith('{} &'):
      new_lines.append(line.replace('{} &', '').replace(
        '{c |}{price}', '{| c |}{price}'))
    elif oneline and line.startswith('0 &'):
      new_lines.append(line.replace('0 &', ''))
    else:
      new_lines.append(line)

  pdf_table = '\n'.join(new_lines)

  pdf_table = pdf_table.replace('nb_stocks', '$d$')
  pdf_table = pdf_table.replace('hidden_size', '$K$')
  pdf_table = pdf_table.replace('nb_epochs', 'epochs')
  pdf_table = pdf_table.replace('use_path', 'use path')
  pdf_table = pdf_table.replace('ridge_coeff', 'ridge coeff')
  pdf_table = pdf_table.replace('train_ITM_only', 'train ITM only')
  pdf_table = pdf_table.replace('nb_dates', '$N$')
  pdf_table = pdf_table.replace('spot', '$x_0$')
  pdf_table = pdf_table.replace('use_payoff_as_input', 'use P')


  with tf.io.gfile.GFile(table_path, "w") as tablef:
    tablef.write(pdf_table)
  print(f"{table_path} written.")
