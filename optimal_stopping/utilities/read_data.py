"""Read data from CSVs according to passed flags."""

import os

import pandas as pd

from optimal_stopping.run import configs
from optimal_stopping.utilities import filtering



INDEX = ["algo", "model", "payoff", "drift", "nb_stocks", "spot", "volatility",
         "nb_paths", "nb_dates", "strike", "dividend", "maturity",
         "hidden_size", "nb_epochs", "hurst",
         "factors", "ridge_coeff", "use_path", "train_ITM_only",
         "use_payoff_as_input",
         ]

old_new_algo_dict = {
  "L": "NLSM",
  "FQIRfast": "RFQI",
  "LNDfast": "RLSM",
  "FQIfast": "FQI",
  "LS": "LSM",
  "DO": "DOS",
  "randRNN": "RRLSM",
  "FQIRfastRNN": "RRFQI"
}

def replace_old_algo_names():
  csv_paths = get_csv_paths() + get_csv_paths_draft()

  for f in csv_paths:
    df = pd.read_csv(f, index_col=None)
    df.replace(to_replace={"algo": old_new_algo_dict}, inplace=True)
    df.to_csv(f, index=False)


def get_csv_paths():
  """Returns a list of CSV files from which to read the data."""
  csvs_dir = os.path.join(os.path.dirname(__file__), "../../output/metrics")
  if not os.path.exists(csvs_dir):
    os.makedirs(csvs_dir)
  return [os.path.join(csvs_dir, fname)
          for fname in sorted(os.listdir(csvs_dir))
          if fname.endswith('.csv')]

def get_csv_paths_draft():
  """Returns a list of CSV files from which to read the data."""
  csvs_dir = os.path.join(os.path.dirname(__file__), "../../output/metrics_draft")
  if not os.path.exists(csvs_dir):
    os.makedirs(csvs_dir)
  return [os.path.join(csvs_dir, fname)
          for fname in sorted(os.listdir(csvs_dir))
          if fname.endswith('.csv')]


def read_csv(path: str, config: configs._DefaultConfig,
             reverse_filtering: bool=False):
  """Reads one CSV and filters out unwanted values."""
  try:
    df = pd.read_csv(path, index_col=INDEX)
  except Exception:
    df = pd.read_csv(path, index_col=None)
    for col in INDEX:
      if col not in df.columns:
        df[col] = None
    df.to_csv(path, index=False)
    df = pd.read_csv(path, index_col=INDEX)

  # print(df)
  return filtering.filter_df(df, config, reverse_filtering)


def read_csvs_conv(which=0):
  if which == 0:
    csv_paths = get_csv_paths_draft()
  elif which == 1:
    csv_paths = get_csv_paths()
  else:
    csv_paths = get_csv_paths_draft() + get_csv_paths()
  df = pd.concat(pd.read_csv(path, index_col=None) for path in csv_paths)
  return df

def read_csvs(config: configs._DefaultConfig, remove_duplicates: bool=True):
  """Returns dataframe with all CSV(s) content, filtered according to config.

  Args:
    config: to filter data using config properties.

  CSVs are read in alpha order (oldest to most recent).
  """
  csv_paths = get_csv_paths() + get_csv_paths_draft()
  print(f"Reading data from {len(csv_paths)} CSV files...")

  df = pd.concat(read_csv(path, config) for path in csv_paths)
  assert  len(df) > 0, "No data read with given filters..."
  if remove_duplicates:
    df = df[~df.index.duplicated(keep='last')]

  return df


def extract_single_value_indexes(df):
  """Returns (df - single value indexes, description of removed params).

  Params:
    df: DataFrame as returned by read_csvs.
  """
  global_params = []
  # params_for_caption = []
  for index_name in df.index.names:
    values = df.index.get_level_values(index_name)
    if len(values.unique()) == 1:
      global_params.append(f"{index_name} = {values[0]}")
      # params_for_caption.append(values[0])
      df = df.reset_index(index_name)
      df = df.drop(columns=index_name)
  global_params_caption = ", ".join(global_params)
  return df, global_params_caption
