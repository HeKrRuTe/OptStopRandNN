import socket
import matplotlib
import matplotlib.colors
import numpy as np
import pandas as pd
import os
import tensorflow as tf

from optimal_stopping.utilities import read_data
from optimal_stopping.utilities import comparison_table as ct
from optimal_stopping.utilities.comparison_table import _human_time_delta


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

dos_t = list(np.linspace(0,1,21))
dos_t[0] = 0.01
dos_p = [1.519, 1.293, 1.049, 0.839, 0.658, 0.503, 0.370, 0.255, 0.156, 0.071,
         0.002, 0.061, 0.117, 0.164, 0.207, 0.244, 0.277, 0.308, 0.337, 0.366,
         0.395]

def plot_hurst(
    filters,
    save_path="../latex/plots/",
    save_extras={'bbox_inches':'tight', 'pad_inches': 0.01},
):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['--', '-.', '-', ':']

    df = read_data.read_csvs_conv(which=0)
    df = df.drop(columns="duration")

    comp_data = []

    hurst = sorted(list(set(df["hurst"].values)))

    f = plt.figure()

    table_dict = {}

    for i, filter in enumerate(filters):
        df_ = df
        for key, val in filter.items():
            if key == "label":
                label = val
            elif key in ["color", "linestyle"]:
                pass
            else:
                df_ = df_.loc[df_[key] == val]
        price = []
        std = []
        times = []
        for h in hurst:
            price.append(np.mean(df_.loc[df_["hurst"] == h, "price"].values))
            std.append(np.std(df_.loc[df_["hurst"] == h, "price"].values))
            times.append(np.median(df_.loc[df_["hurst"] == h, "comp_time"].values))
        table_dict[label] = [
            "{:.2f} ({:.2f})".format(x,y) for x,y in zip(price, std)]
        table_dict["{}_duration".format(label)] = [
            _human_time_delta(sec) for sec in times]
        print("price", len(price), price)
        comp_data.append([filter, np.mean((np.array(dos_p)-np.array(price))**2)])
        if "color" not in filter:
            filter["color"] = colors[i % len(colors)]
        if "linestyle" not in filter:
            filter["linestyle"] = linestyles[i % len(linestyles)]
        # plt.plot(hurst, price, label=label, color=filter["color"],
        #          linestyle=filter["linestyle"])
        plt.errorbar(hurst, price, std, label=label, color=filter["color"],
                     linestyle=filter["linestyle"])

    comp_df = pd.DataFrame(data=comp_data, columns=["desc", "diff"])
    comp_df = comp_df.sort_values(by=["diff"], ascending=True)

    table = pd.DataFrame(data=table_dict, index=hurst)

    plt.plot(dos_t, dos_p, label="pathDOS-paper", color="black")
    plt.xlabel("Hurst")
    plt.ylabel("price")
    plt.legend()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = "{}hurst_plot.png".format(save_path)
    save_comp_file = "{}hurst_comp.csv".format(save_path)
    save_comp_file2 = "{}hurst_table.csv".format(save_path)
    plt.savefig(save_file, **save_extras)
    comp_df.to_csv(save_comp_file)
    table.to_csv(save_comp_file2)

    SBM.send_notification(
        text_for_files='hurst plot',
        chat_id=chat_id,
        files=[save_file, save_comp_file, save_comp_file2],
        text=None,
    )

def get_hurst_table_tex(path):
    df = pd.read_csv(path, index_col=0)
    algos = sorted(list(df.columns.values))[::2]
    columns=["H", "algo", "price", "duration"]
    label = "hurst_table_all"
    column_names = ["price", "duration"]

    table = pd.DataFrame(columns=columns)
    for algo in algos:
        price = df[algo]
        dur = df["{}_duration".format(algo)]
        hurst = df.index.values
        hurst = [str(np.round(x, 3)) for x in hurst]
        table_app = pd.DataFrame(
            data={"H": hurst, "algo": [algo]*len(hurst),
                  "price": price, "duration": dur})
        table = pd.concat([table, table_app], ignore_index=True)

    table.set_index(["H", "algo"], inplace=True)
    df = table.unstack("algo")

    # ========= from here on: copied from comparison_table.py =============
    def my_key(index):
        if index.name == 'algo':
            return pd.Index([ct.ALGOS_ORDER.index(algo) for algo in index], name='algo')
        return pd.Index([ct.COLUMNS_ORDER.index(name) for name in index], name='')
    try:
        df = df.sort_index(key=my_key, axis='columns')
    except Exception:
        df = df.sort_index(key=my_key)
        df = df.to_frame().T
    print(df)
    algos = df.columns.get_level_values("algo").unique()

    # bold_algos = ["RLSM", "RRLSM", "RFQI"]
    bold_algos = []
    _table_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"../../../latex/tables_draft/"))
    if not os.path.exists(_table_path):
        os.makedirs(_table_path)
    table_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"../../../latex/tables_draft/{label}.tex"))
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
    pdf_table = ct._PDF_TABLE_TMPL % {
        "table": df.to_latex(
            na_rep="-", multirow=True, multicolumn=True,
            multicolumn_format='c |',
            float_format="%.2f",
            column_format=col_format),
        "caption": "",
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
                or line.startswith('model ') or line.startswith('H '):
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



if __name__ == '__main__':

    filters1 = [
        {"label": "pathDOS", "algo": "DOS", "color": "orange",
         "linestyle": "--",
         "model": "FractionalBrownianMotionPathDep", 'train_ITM_only': False},
        {"label": "DOS", "algo": "DOS", "model": "FractionalBrownianMotion",
         "linestyle": "-",
         'train_ITM_only': False, "color": "gray",},
        {"label": "RLSM", "algo": "RLSM", "color": "red",
         "linestyle": "-.",
         "model": "FractionalBrownianMotion", 'train_ITM_only': False},
        {"label": "RRLSM", "algo": "RRLSM", "color": "blue",
         "linestyle": ":", "nb_paths": 20000,
         "model": "FractionalBrownianMotion", 'train_ITM_only': False},

        # {"label": "LL", "algo": "L", "model": "FractionalBrownianMotion",
        #  'train_ITM_only': False},
        # {"label": "pathLL", "algo": "L",
        #  "model": "FractionalBrownianMotionPathDep", 'train_ITM_only': False},
        # {"label": "FQIR", "algo": "FQIRfast", "model": "FractionalBrownianMotion",
        #  'train_ITM_only': False},
        # {"label": "pathFQIR", "algo": "FQIRfast",
        #  "model": "FractionalBrownianMotionPathDep",
        #  'train_ITM_only': False},
        # {"label": "LS", "algo": "LS", "model": "FractionalBrownianMotion",
        #  'train_ITM_only': False},
        # {"label": "pathLS", "algo": "LS",
        #  "model": "FractionalBrownianMotionPathDep", 'train_ITM_only': False},
    ]
    plot_hurst(filters=filters1)

    filters2 = [
        {"label": "FQI", "algo": "FQI",  "color": "pink",
         "linestyle": "-.",
         "model": "FractionalBrownianMotion", 'train_ITM_only': False},
        {"label": "RFQI", "algo": "RFQI", "color": "olive",
         "linestyle": "-",
         "model": "FractionalBrownianMotion", 'train_ITM_only': False},
        {"label": "pathRFQI", "algo": "RFQI", "color": "purple",
         "linestyle": "--",
         "model": "FractionalBrownianMotionPathDep", 'train_ITM_only': False},
        {"label": "RRFQI", "algo": "RRFQI",  "color": "brown",
         "linestyle": ":",
         "model": "FractionalBrownianMotion", 'train_ITM_only': False},
    ]
    plot_hurst(filters=filters2)

    filters3 = [
        {"label": "RRLSM", "algo": "RRLSM", "color": "blue",
         "linestyle": ":", "nb_paths": 20000,
         "model": "FractionalBrownianMotion", 'train_ITM_only': False},
    ]
    plot_hurst(filters=filters3)

    filters4 = [
        {"label": "RRLSM", "algo": "RRLSM", "color": "blue",
         "linestyle": ":", "nb_paths": 20000,
         "model": "FractionalBrownianMotion", 'train_ITM_only': False},
        {"label": "RRLSM", "algo": "RRLSM", "color": "orange",
         "linestyle": "--", "nb_paths": 400000,
         "model": "FractionalBrownianMotion", 'train_ITM_only': False},
    ]
    # plot_hurst(filters=filters4)

    get_hurst_table_tex("/Users/fkrach/ETH/PhD/Research/Calypso/amc2/latex/plots/hurst_table_all.csv")


