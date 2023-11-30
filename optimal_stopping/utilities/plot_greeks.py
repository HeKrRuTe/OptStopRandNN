"""
author: Florian Krach
"""


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



def plot_greeks(
        config: configs._DefaultConfig,
        greeks_method="regression",
        algo="RLSMSoftplus",
        volatilities=(0.1, 0.2, 0.3, 0.4),
        maturities=(0.5, 1, 2, 4, 8),
        save_path="../latex/plots/",
        save_extras={'bbox_inches':'tight', 'pad_inches': 0.01},
):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['-', '-.', '--', ':']

    df = read_data.read_csvs_conv(which=0)
    df = df.drop(columns="duration")
    df = df.loc[df["algo"] == algo]
    df = df.loc[df["greeks_method"] == greeks_method]
    df = df.loc[df["volatility"].isin(volatilities)]
    df = df.loc[df["maturity"].isin(maturities)]


    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    spots = sorted(list(set(df["spot"].values)))

    for n, plot_label in enumerate(
            ["price", "delta", "gamma", "theta", "rho", "vega"]):
        j = n % 3
        i = n // 3
        ax = axs[i,j]

        for v in volatilities:
            for m in maturities:
                if len(df.loc[((df["volatility"] == v) &
                               (df["maturity"] == m))]) == 0:
                    continue
                vals = []
                for s in spots:
                    _df = df.loc[((df["volatility"] == v) &
                                (df["maturity"] == m) &
                                (df["spot"] == s)), plot_label]
                    which = _df.abs() > 500
                    printdf = _df.loc[which]
                    if len(printdf) > 0:
                        print(printdf)
                    vals.append(_df.median())
                ax.plot(spots, vals, label="$\\sigma=${:.1f} $T=${:.1f}".format(v, m))
        ax.set_title(plot_label)

    plt.legend(bbox_to_anchor=(1.04, 1.1), loc="center left")
    plt.subplots_adjust(right=0.75)
    plt.savefig(
        "{}greeks_plot_{}_{}.pdf".format(save_path, algo, greeks_method),
        **save_extras)














if __name__ == '__main__':

    plot_greeks(
        config=configs.table_greeks_plots,
        greeks_method="regression",
        algo="RLSMSoftplus",
        volatilities=(0.1, 0.2, 0.3),
        maturities=(0.5, 1, 2,),
    )

    # plot_greeks(
    #     config=configs.table_greeks_plots_binomial,
    #     greeks_method="central",
    #     algo="B",
    #     volatilities=(0.1, 0.2, 0.3),
    #     maturities=(0.5, 1, 2,), )


    pass
