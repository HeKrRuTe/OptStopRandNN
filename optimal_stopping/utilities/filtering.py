"""Module to filter DataFrame based on config.
"""
from absl import flags

from optimal_stopping.run import configs

flags.DEFINE_bool(
    "debug_csv_filtering", False,
    "Set to True to display why rows are filtered out and a snippet of data."
)


FILTERS = [
    # Config attr name, index name.
    ("algos", "algo"),
    ("payoffs", "payoff"),
    ("dividends", "dividend"),
    ("spots", "spot"),
    ("volatilities", "volatility"),
    ("maturities", "maturity"),
    ("nb_paths", "nb_paths"),
    ("nb_dates", "nb_dates"),
    ("nb_stocks", "nb_stocks"),
    ("drift", "drift"),
    ("stock_models", "model"),
    ("strikes", "strike"),
    ("hurst", "hurst"),
    ("hidden_size", "hidden_size"),
    ("factors", "factors"),
    ("ridge_coeff", "ridge_coeff"),
    ("use_payoff_as_input", "use_payoff_as_input"),
]

FLAGS = flags.FLAGS


def filter_df(df, config: configs._DefaultConfig,
              reverse_filtering: bool=False):
  """Returns new DataFrame with rows removed according to passed config.

  Args:
    reverse_filtering: bool, default to False. If True, inverse the selection.
    ie: --algos=LN will return all algos but LN.
  """
  for filter_name, column_name in FILTERS:
    values = list(getattr(config, filter_name))
    if filter_name == "factors":
        values = [str(x) for x in values]
    idx = df.index.get_level_values(column_name).isin(values)
    if reverse_filtering:
      idx = ~idx
    if FLAGS.debug_csv_filtering and any(~idx):
      print(f"Filtering out {sum(~idx)} rows because {column_name} "
            f"not in {values}")
      print(df[~idx])
    df = df[idx]
  return df
