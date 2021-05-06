"""Write PDF table to compare algorithms results."""
import os.path
from typing import Iterable

import pandas as pd
import tensorflow as tf

from optimal_stopping.run import configs
from optimal_stopping.utilities import read_data


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
  "LSPI", "LSM", "LSMRidge",  "LSMLaguerre",
  "DOS", "pathDO", "NLSM",
  "RLSM", "RLSMRidge", "RRLSM",
  "FQI", "FQIR", "FQILaguerre",
  "RFQI", "RRFQI",
]
COLUMNS_ORDER = ["price", "duration"]


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
        get_df=False
):
  df = read_data.read_csvs(config, remove_duplicates=False)
  df = df.filter(items=column_names)

  if 'price' in column_names:
    mean_price = df.groupby(df.index)['price'].mean()
    std = df.groupby(df.index)['price'].std()
    #size = df.groupby(df.index)['price'].size()[0]
    #conf_interval_price = 1.96 * std  / math.sqrt(size)

  if 'duration' in column_names:
    median_duration = df.groupby(df.index)['duration'].median()

  df = df[~df.index.duplicated(keep='last')]

  print(df)
  if 'price' in column_names:
    if get_df:
      df['price'] = mean_price
    else:
      df['mean_price'] = mean_price
      df['std_price'] = std #conf_interval_price
      df['price'] = ['%.2f (%.2f)' % ms
                     for ms in zip(df['mean_price'], df['std_price'])]
      df = df.drop('mean_price', 'columns')
      df = df.drop('std_price', 'columns')

  if 'duration' in column_names:
    try:
      df['duration'] = median_duration
      df['duration'] = [_human_time_delta(sec)
                             for sec in df['duration']]
    except Exception:
      df['duration'] = None

  df, global_params_caption = read_data.extract_single_value_indexes(df)
  # print(df)
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

  if get_df:
    return df

  algos = df.columns.get_level_values("algo").unique()

  bold_algos = ["RLSM", "RRLSM", "RFQI"]
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
        or line.startswith('maturity &') or line.startswith('payoff '):
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


  with tf.io.gfile.GFile(table_path, "w") as tablef:
    tablef.write(pdf_table)
  print(f"{table_path} written.")
