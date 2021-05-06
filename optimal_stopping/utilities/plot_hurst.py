import socket
import matplotlib
import matplotlib.colors
import numpy as np
import pandas as pd
import os

from optimal_stopping.utilities import read_data


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
        for h in hurst:
            price.append(np.mean(df_.loc[df_["hurst"] == h, "price"].values))
        # print("price", len(price), price)
        comp_data.append([filter, np.mean((np.array(dos_p)-np.array(price))**2)])
        if "color" not in filter:
            filter["color"] = colors[i % len(colors)]
        if "linestyle" not in filter:
            filter["linestyle"] = linestyles[i % len(linestyles)]
        plt.plot(hurst, price, label=label, color=filter["color"],
                 linestyle=filter["linestyle"])

    comp_df = pd.DataFrame(data=comp_data, columns=["desc", "diff"])
    comp_df = comp_df.sort_values(by=["diff"], ascending=True)

    plt.plot(dos_t, dos_p, label="pathDOS-paper", color="black")
    plt.xlabel("hurst")
    plt.ylabel("price")
    plt.legend()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = "{}hurst_plot.png".format(save_path)
    save_comp_file = "{}hurst_comp.csv".format(save_path)
    plt.savefig(save_file, **save_extras)
    comp_df.to_csv(save_comp_file)

    SBM.send_notification(
        text_for_files='hurst plot',
        chat_id=chat_id,
        files=[save_file, save_comp_file],
        text=None,
    )



if __name__ == '__main__':

    filters1 = [
        {"label": "pathDOS", "algo": "DO", "color": "orange",
         "linestyle": "--",
         "model": "FractionalBrownianMotionPathDep", 'train_ITM_only': False},
        {"label": "DOS", "algo": "DO", "model": "FractionalBrownianMotion",
         "linestyle": "-",
         'train_ITM_only': False, "color": "gray",},
        {"label": "RLSM", "algo": "LNDfast", "color": "red",
         "linestyle": "-.",
         "model": "FractionalBrownianMotion", 'train_ITM_only': False},
        {"label": "RRLSM", "algo": "randRNN", "color": "blue",
         "linestyle": ":",
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
        {"label": "RFQI", "algo": "FQIRfast", "color": "olive",
         "linestyle": "-",
         "model": "FractionalBrownianMotion", 'train_ITM_only': False},
        {"label": "pathRFQI", "algo": "FQIRfast", "color": "purple",
         "linestyle": "--",
         "model": "FractionalBrownianMotionPathDep", 'train_ITM_only': False},
        {"label": "RRFQI", "algo": "FQIRfastRNN",  "color": "brown",
         "linestyle": ":",
         "model": "FractionalBrownianMotion", 'train_ITM_only': False},
    ]
    plot_hurst(filters=filters2)

    filters3 = [
        {"label": "RRLSM", "algo": "randRNN", "color": "blue",
         "linestyle": ":",
         "model": "FractionalBrownianMotion", 'train_ITM_only': False},
    ]
    plot_hurst(filters=filters3)


