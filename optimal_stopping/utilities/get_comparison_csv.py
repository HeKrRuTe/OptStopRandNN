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

def get_comparison_csv(
    filters,
    save_path="../latex/plots/",
):
    df = read_data.read_csvs_conv(which=0)
    df = df.drop(columns="duration")

    comp_data = []

    for filter in filters:
        df_ = df
        for key, val in filter.items():
            if key == "label":
                label = val
            else:
                df_ = df_.loc[df_[key] == val]
        price = np.mean(df_["price"].values)
        std = np.std(df_["price"].values)
        comp_data.append([filter, price, std])

    comp_df = pd.DataFrame(data=comp_data, columns=["desc", "price", "std"])
    comp_df = comp_df.sort_values(by=["price"], ascending=False)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_comp_file = "{}comparison.csv".format(save_path)
    comp_df.to_csv(save_comp_file)

    SBM.send_notification(
        text_for_files='comparison csv',
        chat_id=chat_id,
        files=[save_comp_file],
        text=None,
    )



if __name__ == '__main__':

    filters4 = []
    factors = [[0.00005], [0.0001], [0.0002]]
    for f in factors:
        filters4 += [
            {"algo": "LNDfastTanh",
             "model": "BlackScholes",
             "factors": "{}".format(f)},
        ]
    factors = []
    for a in [0.00005, 0.0001, 0.0002]:
        for b in [0.05, 0.1, 0.15, 0.2]:
            factors += [[a, b]]
    for f in factors:
        filters4 += [
            {"algo": "randRNN",
             "model": "BlackScholes",
             "factors": "{}".format(f)},
        ]
    filters4 += [{"algo": "FQIRfast", "model": "BlackScholes"},]
    # get_comparison_csv(filters=filters4)


    factors = []
    for a in [0.2, 0.5, 1, 1.5, 2, 4]:
        factors += [[a]]
    filters = [{"algo": "FQIRfast", "model": "BlackScholes"}]
    for f in factors:
        filters += [
            {"algo": "LNDfast",
             "model": "BlackScholes",
             "factors": "{}".format(f)},
            {"algo": "FQIRfast",
             "model": "BlackScholes",
             "factors": "{}".format(f)},
        ]
    # get_comparison_csv(filters=filters)

    factors3 = []
    for a in [0.0001, 0.0005, 0.001, 0.005]:
        for b in [0.005, 0.01, 0.05, 0.1, 0.2, 0.3]:
            factors3 += [[a, b]]
    filters = []
    for f in factors3:
        for h in [0.05]:
            for s in [5, 10, ]:
                for p in ['Max', 'Mean']:
                    filters += [
                        {"algo": "randRNN",
                         "hurst": h,
                         "payoff": p,
                         "nb_stocks": s,
                         "model": "FractionalBrownianMotion",
                         "nb_paths": 20000,
                         "factors": "{}".format(f)},
                    ]
    get_comparison_csv(filters)


