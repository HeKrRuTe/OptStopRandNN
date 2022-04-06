"""
author: Florian Krach
"""
import socket
import matplotlib
import matplotlib.colors
import numpy as np
import pandas as pd
import os
from absl import app
import itertools

from optimal_stopping.utilities import read_data
from optimal_stopping.run import configs
from optimal_stopping.utilities import configs_getter
from optimal_stopping.utilities.comparison_table import USE_PAYOFF_FOR_ALGO

# TODO: change for use payoff or not


if 'ada-' in socket.gethostname() or 'arago' in socket.gethostname():
    SERVER = True
else:
    SERVER = False


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



def plot_tables(*args, **kwargs):
    which_time = "comp_time"
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['--', '-.', '-', ':']
    save_extras={'bbox_inches':'tight', 'pad_inches': 0.01}

    for config_name, config in configs_getter.get_configs():
        df = read_data.read_csvs(config, remove_duplicates=False)

        # model = config.stock_models
        # assert len(model) == 1
        # model = model[0]

        index = ["algo", "payoff", "nb_paths", "nb_dates", "spot", "nb_stocks",
                 "model", "use_payoff_as_input"]
        nbstocks_i = np.where(np.array(index) == "nb_stocks")[0][0]
        nbpaths_i = np.where(np.array(index) == "nb_paths")[0][0]
        payoff_i = np.where(np.array(index) == "payoff")[0][0]
        nbdates_i = np.where(np.array(index) == "nb_dates")[0][0]
        algos_i = np.where(np.array(index) == "algo")[0][0]
        spot_i = np.where(np.array(index) == "spot")[0][0]
        model_i = np.where(np.array(index) == "model")[0][0]
        upayoff_i = np.where(np.array(index) == "use_payoff_as_input")[0][0]

        df.reset_index(inplace=True)
        df.replace(to_replace=np.nan, value="no_val", inplace=True)
        df.set_index(index, inplace=True)
        df = df.filter(items=["price", which_time])


        mean_price = df.groupby(df.index)['price'].mean()
        std = df.groupby(df.index)['price'].std()
        median_duration = df.groupby(df.index)[which_time].median()

        all_nbstocks = sorted(list(set([x[nbstocks_i] for x in mean_price.index])))
        all_nbpaths = sorted(list(set([x[nbpaths_i] for x in mean_price.index])))
        all_payoffs = sorted(list(set([x[payoff_i] for x in mean_price.index])))
        all_nbdates = sorted(list(set([x[nbdates_i] for x in mean_price.index])))
        all_algos = sorted(list(set([x[algos_i] for x in mean_price.index])))
        all_spots = sorted(list(set([x[spot_i] for x in mean_price.index])))
        all_models = sorted(list(set([x[model_i] for x in mean_price.index])))

        # sort B and EOP to end
        for a in ["B", "EOP"]:
            if a in all_algos:
                all_algos.remove(a)
                all_algos += [a]

        print(all_nbstocks)
        print(all_nbpaths)
        print(all_payoffs)
        print(all_nbdates)
        print(all_algos)
        print(all_spots)

        # plot different combinations in extra plots
        for nbpaths, payoff, nbdates, spot, model in itertools.product(
                all_nbpaths, all_payoffs, all_nbdates, all_spots, all_models):
            # plot each algo
            fig, ax = plt.subplots(2,1)

            # first plot only the ML algos to get plot axis
            algos_ = []
            for algo in all_algos:
                if algo not in ["FQI", "LSM"]:
                    algos_.append(algo)
            for i, algo in enumerate(algos_):
                prices = []
                err = []
                stocks = []
                time = []
                for d in all_nbstocks:
                    try:
                        prices.append(mean_price[(
                            algo, payoff, nbpaths, nbdates, spot, d, model,
                            USE_PAYOFF_FOR_ALGO[algo])])
                        err.append(std[(
                            algo, payoff, nbpaths, nbdates, spot, d, model,
                            USE_PAYOFF_FOR_ALGO[algo])])
                        time.append(median_duration[(
                            algo, payoff, nbpaths, nbdates, spot, d, model,
                            USE_PAYOFF_FOR_ALGO[algo])])
                        stocks.append(d)
                    except ValueError:
                        pass
                color = colors[i % len(colors)]
                lst = linestyles[i % len(linestyles)]
                if len(prices) > 0:
                    ax[0].errorbar(
                        x=stocks, y=prices, yerr=err, label=algo, color=color,
                        linestyle=lst)
                    ax[1].plot(
                        stocks, time, label=algo, color=color, linestyle=lst)
                else:
                    all_algos.remove(algo)

            # now make the real plot
            ylim0 = ax[0].get_ylim()
            ylim1 = ax[1].get_ylim()
            fig, ax = plt.subplots(2,1)
            for i, algo in enumerate(all_algos):
                prices = []
                err = []
                stocks = []
                time = []
                for d in all_nbstocks:
                    try:
                        prices.append(mean_price[(
                            algo, payoff, nbpaths, nbdates, spot, d, model,
                            USE_PAYOFF_FOR_ALGO[algo])])
                        err.append(std[(
                            algo, payoff, nbpaths, nbdates, spot, d, model,
                            USE_PAYOFF_FOR_ALGO[algo])])
                        time.append(median_duration[(
                            algo, payoff, nbpaths, nbdates, spot, d, model,
                            USE_PAYOFF_FOR_ALGO[algo])])
                        stocks.append(d)
                    except ValueError:
                        pass
                color = colors[i % len(colors)]
                lst = linestyles[i % len(linestyles)]
                if algo in ["B", "EOP"]:
                    color = "black"
                    lst = "-"
                if len(prices) > 0:
                    ax[0].errorbar(
                        x=stocks, y=prices, yerr=err, label=algo, color=color,
                        linestyle=lst)
                    ax[1].plot(
                        stocks, time, label=algo, color=color, linestyle=lst)

            # ax[1].set_yscale("log")
            # ax[0].set_xscale("log")
            # ax[1].set_xscale("log")
            ax[0].set_ylim(ylim0)
            ax[1].set_ylim(ylim1)
            plt.legend()
            plt.xlabel("number of stocks $d$")
            ax[0].set_ylabel("price")
            ax[1].set_ylabel("time (s)")
            # plt.show()
            path = "../latex/plots/{}-{}-{}-{}-{}.pdf".format(model, nbpaths, payoff, nbdates, spot)
            plt.savefig(path, **save_extras)



if __name__ == '__main__':
    app.run(plot_tables)
    pass
