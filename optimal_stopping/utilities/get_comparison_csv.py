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
    save_path="../../../latex/plots/",
    sort=True,
    filters_to_add_as_columns=[],
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
        if len(df_) > 0:
            price = np.mean(df_["price"].values)
            std = np.std(df_["price"].values)
            try:
                delta = np.mean(df_["delta"].values)
                gamma = np.mean(df_["gamma"].values)
                theta = np.mean(df_["theta"].values)
                rho = np.mean(df_["rho"].values)
                vega = np.mean(df_["vega"].values)
                if len(df_) > 1:
                    delta_std = np.std(df_["delta"].values)
                    gamma_std = np.std(df_["gamma"].values)
                    theta_std = np.std(df_["theta"].values)
                    rho_std = np.std(df_["rho"].values)
                    vega_std = np.std(df_["vega"].values)
                    price_w = "{:.4f} ({:.4f})".format(price, std)
                    delta_w = "{:.4f} ({:.4f})".format(delta, delta_std)
                    gamma_w = "{:.4f} ({:.4f})".format(gamma, gamma_std)
                    theta_w = "{:.4f} ({:.4f})".format(theta, theta_std)
                    rho_w = "{:.4f} ({:.4f})".format(rho, rho_std)
                    vega_w = "{:.4f} ({:.4f})".format(vega, vega_std)
                else:
                    delta_std, gamma_std, theta_std, rho_std, \
                    vega_std = [None]*5
                    price_w = "{:.4f} (--)".format(price)
                    delta_w = "{:.4f} (--)".format(delta)
                    gamma_w = "{:.4f} (--)".format(gamma)
                    theta_w = "{:.4f} (--)".format(theta)
                    rho_w = "{:.4f} (--)".format(rho)
                    vega_w = "{:.4f} (--)".format(vega)
                if len(price_w) > 20:
                    price_w = "--"
                if len(delta_w) > 20:
                    delta_w = "--"
                if len(gamma_w) > 20:
                    gamma_w = "--"
                if len(theta_w) > 20:
                    theta_w = "--"
                if len(rho_w) > 20:
                    rho_w = "--"
                if len(vega_w) > 20:
                    vega_w = "--"
            except Exception:
                delta, delta_std, gamma, gamma_std, theta, theta_std, \
                rho, rho_std, vega, vega_std = [None]*10
                price_w, delta_w, gamma_w, theta_w, rho_w, vega_w = [None]*6
            _comp_data = [filter]
            for fi in filters_to_add_as_columns:
                try:
                    _comp_data.append(filter[fi])
                except Exception:
                    _comp_data.append(None)
            _comp_data += [price, std, delta, delta_std, gamma, gamma_std,
                           theta, theta_std, rho, rho_std, vega, vega_std,
                           price_w, delta_w, gamma_w, theta_w, rho_w, vega_w]
            comp_data.append(_comp_data)

    cols = ["desc"] + filters_to_add_as_columns + \
           ["price", "std", "delta", "delta_std", "gamma", "gamma_std",
            "theta", "theta_std", "rho", "rho_std", "vega", "vega_std",
            "price_w", "delta_w", "gamma_w", "theta_w", "rho_w", "vega_w"]
    comp_df = pd.DataFrame(
        data=comp_data,
        columns=cols)
    if sort:
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


def combine_FD_regr_greeks_table(
        save_path="../../../latex/plots/", filters_to_add_as_columns=[]):
    save_comp_file = "{}comparison.csv".format(save_path)
    df = pd.read_csv(save_comp_file)
    cols = ["price_w", "delta_w", "gamma_w", "theta_w", "rho_w", "vega_w"]
    df = df[filters_to_add_as_columns+cols]
    df1 = df.loc[df["greeks_method"] == "central"]
    df2 = df.loc[df["greeks_method"] == "regression"]
    df1 = df1[filters_to_add_as_columns+cols]
    df2 = df2[filters_to_add_as_columns+["price_w", "delta_w", "gamma_w"]]
    df2.rename(columns={
        "price_w": "price_w_reg", "delta_w": "delta_w_reg",
        "gamma_w": "gamma_w_reg"}, inplace=True)
    df1.drop(columns=["greeks_method"], inplace=True)
    df2.drop(columns=["greeks_method"], inplace=True)
    filters_to_add_as_columns.remove("greeks_method")
    df3 = pd.merge(df1, df2, on=filters_to_add_as_columns, how="left")
    df3.rename(columns=lambda s: s.replace("_", ""), inplace=True)
    df3.replace("RLSMSoftplus", "RLSM", inplace=True)
    save_comp_file = "{}greeks_comparison.csv".format(save_path)
    df3.to_csv(save_comp_file, index=False)
    if "strike" in df3.columns:
        strikes = sorted(list(set(df3["strike"].values)))
        for i, s in enumerate(strikes):
            save_comp_file_ = "{}greeks_comparison{}.csv".format(save_path, i+1)
            df4 = df3.loc[df3["strike"] == s]
            df4.to_csv(save_comp_file_, index=False)


if __name__ == '__main__':
    filter_greeks = []
    for strike in [36, 40, 44]:
        filter_greeks += [{"algo": "B",
                           "strike": strike,
                           "greeks_method": "central",
                           "nb_dates": 50000,
                           "model": "BlackScholes",},]
        for method in ["central", "regression"]:
            for algo in ["LSM", "RLSMSoftplus", "FQI", "RFQI", "NLSM"]:
                filter_greeks += [
                    {"algo": algo,
                     "strike": strike,
                     "greeks_method": method,
                     "use_payoff_as_input": False,
                     "nb_epochs": 10,
                     "model": "BlackScholes",},
                ]
            filter_greeks += [
                {"algo": "DOS",
                 "strike": strike,
                 "greeks_method": method,
                 "use_payoff_as_input": True,
                 "nb_epochs": 10,
                 "model": "BlackScholes",},
            ]

    filters_to_add_as_columns = ["algo", "strike", "greeks_method"]
    get_comparison_csv(
        filter_greeks, sort=False,
        filters_to_add_as_columns=filters_to_add_as_columns)
    combine_FD_regr_greeks_table(
        filters_to_add_as_columns=filters_to_add_as_columns)




