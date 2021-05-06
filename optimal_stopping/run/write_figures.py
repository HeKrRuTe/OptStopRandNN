"""Writes all figures and PDF, based on configs and available data."""
import atexit
import os

from absl import app

from optimal_stopping.utilities import configs_getter
from optimal_stopping.utilities import comparison_table
from optimal_stopping.utilities import plot_hurst
from optimal_stopping.utilities import plot_convergence_study


def write_figures():
  for config_name, config in configs_getter.get_configs():
    representations = list(config.representations)
    print(config_name, config, representations)
    for representation in representations:
      if len(representations) > 1:
        figure_name = f"{config_name}_{representation}"
      else:
        figure_name = config_name
      print(f"Writing {config_name}...")
      methods = {
          "TablePrice": comparison_table.write_table_price,
          "TableDuration": comparison_table.write_table_duration,
          "TablePriceDuration": comparison_table.write_table_price_duration,
          "ConvergenceStudy": plot_convergence_study.plot_convergence_study,
          "PlotHurst": plot_hurst.plot_hurst,
      }
      if representation not in methods:
        raise AssertionError(
            f"Unknown representation type {config.representation}")
      try:
        if representation == "ConvergenceStudy":
          methods[representation](config, x_axis="nb_paths")
          methods[representation](config, x_axis="hidden_size")
        else:
          methods[representation](figure_name, config)
      except BaseException as err:
        print("Error:", err)
        raise


def generate_pdf():
  cmd = "pdflatex -synctex=1 -interaction=nonstopmode amc2.tex > /dev/null"
  wd = os.getcwd()
  latex_dir_path = os.path.join(os.path.dirname(__file__), "../../../latex")
  os.chdir(latex_dir_path)
  atexit.register(os.chdir, wd)
  status = os.system(cmd)
  if status == 0:
    pdf_path = os.path.abspath(os.path.join(latex_dir_path, "amc2.pdf"))
    print(f"{pdf_path} written")
  else:
    print("Error while generating amc2.pdf")


def main(argv):
  del argv
  write_figures()
  generate_pdf()


if __name__ == "__main__":
  app.run(main)
