import socket
import matplotlib
import matplotlib.colors
import numpy as np
import pandas as pd
import os

from optimal_stopping.run import configs
from optimal_stopping.utilities import read_data

if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True

if SERVER:
    SEND = True
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

chat_id = "-399803347"



def plot_convergence_study(
        config: configs._DefaultConfig,
        x_axis="nb_paths",
        x_log=True, y_log=False,
        save_path="../latex/plots/",
        save_extras={'bbox_inches':'tight', 'pad_inches': 0.01},
):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['-', '-.', '--', ':']

    df = read_data.read_csvs_conv(which=0)
    df = df.drop(columns="duration")
    df = df.loc[df["algo"].isin(config.algos)]
    df = df.loc[df["model"].isin(config.stock_models)]

    # get sets of network size and training size
    n_sizes = sorted(list(set(df["hidden_size"].values)))
    t_sizes = sorted(list(set(df["nb_paths"].values)))
    if x_axis == "nb_paths":
        x_axis_params = t_sizes
        other_param_name = "hidden_size"
        other_params = n_sizes
        x_axis_name = "number of paths"
    else:
        x_axis = "hidden_size"
        x_axis_params = n_sizes
        other_param_name = "nb_paths"
        other_params = t_sizes
        x_axis_name = "hidden size"

    # get means and stds
    means = []
    stds = []
    for val2 in other_params:
        _means = []
        _stds = []
        for val1 in x_axis_params:
            current_prices = df.loc[
                (df[x_axis] == val1) & (df[other_param_name] == val2), "price"
            ].values
            _means.append(np.mean(current_prices))
            _stds.append(np.std(current_prices))
        means.append(_means)
        stds.append(_stds)

    # plotting
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    for i, args in enumerate(zip(means, stds, other_params)):
        mean, std, val2 = args
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        ax.errorbar(
            x_axis_params, mean, yerr=std,
            ecolor=color, capsize=4, capthick=1, marker=".",
            color=color, linestyle=linestyle,
        )
        ax.errorbar(
            x_axis_params, mean, yerr=std,
            label="{}={}".format(other_param_name.replace('_', ' '), val2),
            ecolor="black", capsize=0, capthick=0, marker=".",
            color=color, linestyle=linestyle,
        )
    plt.xlabel(x_axis_name)
    plt.ylabel("price")
    plt.legend()
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = "{}convergence_plot_{}.png".format(save_path, x_axis)
    plt.savefig(save_file, **save_extras)

    SBM.send_notification(
        text_for_files='convergence plot',
        chat_id=chat_id,
        files=[save_file],
        text=None,
    )
