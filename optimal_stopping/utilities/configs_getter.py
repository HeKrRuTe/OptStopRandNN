import inspect

from absl import flags

from optimal_stopping.run import configs

FLAGS = flags.FLAGS

flags.DEFINE_list("configs", [], "Name of the configs to run.")


def get_configs():
  """Returns (config_name, config) tuples."""
  all_configs = [
      (name, conf) for name, conf in inspect.getmembers(configs)
      if isinstance(conf, configs._DefaultConfig) and not name.startswith('_')]
  if not FLAGS.configs:
    return all_configs
  all_names = [n for n, c in all_configs]
  unknown_requested_configs = set(FLAGS.configs) - set(all_names)
  if unknown_requested_configs:
    raise AssertionError(
        f"--configs flag contain invalid values: {unknown_requested_configs}")
  return [(name, config) for name, config in all_configs
          if name in FLAGS.configs]
