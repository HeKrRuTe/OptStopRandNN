import socket
import matplotlib
import matplotlib.colors
import numpy as np
import pandas as pd
import os, io

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

dos_table = """
nb_stocks spot dos_lower timel dos_upper timeu dos_mid confint lit
2 90 8.072 28.7 8.075 25.4 8.074 [8.060,8.081] 8.075
2 100 13.895 28.7 13.903 25.3 13.899 [13.880,13.910] 13.902
2 110 21.353 28.4 21.346 25.3 21.349 [21.336,21.354] 21.345
3 90 11.290 28.8 11.283 26.3 11.287 [11.276,11.290] 11.29
3 100 18.690 28.9 18.691 26.4 18.690 [18.673,18.699] 18.69
3 110 27.564 27.6 27.581 26.3 27.573 [27.545,27.591] 27.58
5 90 16.648 27.6 16.640 28.4 16.644 [16.633,16.648] [16.620,16.653]
5 100 26.156 28.1 26.162 28.3 26.159 [26.138,26.174] [26.115,26.164]
5 110 36.766 27.7 36.777 28.4 36.772 [36.745,36.789] [36.710,36.798]
10 90 26.208 30.4 26.272 33.9 26.240 [26.189,26.289]
10 100 38.321 30.5 38.353 34.0 38.337 [38.300,38.367]
10 110 50.857 30.8 50.914 34.0 50.886 [50.834,50.937]
20 90 37.701 37.2 37.903 44.5 37.802 [37.681,37.942]
20 100 51.571 37.5 51.765 44.3 51.668 [51.549,51.803]
20 110 65.494 37.3 65.762 44.4 65.628 [65.470,65.812]
30 90 44.797 45.1 45.110 56.2 44.953 [44.777,45.161]
30 100 59.498 45.5 59.820 56.3 59.659 [59.476,59.872]
30 110 74.221 45.3 74.515 56.2 74.368 [74.196,74.566]
50 90 53.903 58.7 54.211 79.3 54.057 [53.883,54.266]
50 100 69.582 59.1 69.889 79.3 69.736 [69.560,69.945]
50 110 85.229 59.0 85.697 79.3 85.463 [85.204,85.763]
100 90 66.342 95.5 66.771 147.7 66.556 [66.321,66.842]
100 100 83.380 95.9 83.787 147.7 83.584 [83.357,83.862]
100 110 100.420 95.4 100.906 147.7 100.663 [100.394,100.989]
200 90 78.993 170.9 79.355 274.6 79.174 [78.971,79.416]
200 100 97.405 170.1 97.819 274.3 97.612 [97.381,97.889]
200 110 115.800 170.6 116.377 274.5 116.088 [115.774,116.472]
500 90 95.956 493.4 96.337 761.2 96.147 [95.934,96.407]
500 100 116.235 493.5 116.616 761.7 116.425 [116.210,116.685]
500 110 136.547 493.7 136.983 761.4 136.765 [136.521,137.064]
"""


def get_comparison_csv(
        filters,
        save_path="../latex/plots/",
        sort=True, sortby=["price"], sort_ascending=False,
        filters_to_add_as_columns=[], read_from_which=0,
):
    df = read_data.read_csvs_conv(which=read_from_which)
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
            comp_time = np.median(df_["comp_time"].values)
            if len(df_) > 1:
                price_w = "{:.4f} ({:.4f})".format(price, std)
            else:
                price_w = "{:.4f} (--)".format(price)
            time_w = "{:.2f}s".format(comp_time)
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
                delta_w, gamma_w, theta_w, rho_w, vega_w = [None]*5
            try:
                price_upper = np.mean(df_["price_upper_bound"].values)
                std_upper = np.std(df_["price_upper_bound"].values)
                midprice = np.mean(
                    (df_["price_upper_bound"].values + df_["price"].values)/2)
                std_midprice = np.std(
                    (df_["price_upper_bound"].values + df_["price"].values)/2)
                if len(df_) > 1:
                    price_upper_w = "{:.4f} ({:.4f})".format(
                        price_upper, std_upper)
                    midprice_w = "{:.4f} ({:.4f})".format(
                        midprice, std_midprice)
                else:
                    price_upper_w = "{:.4f} (--)".format(price_upper)
                    midprice_w = "{:.4f} (--)".format(midprice)
            except Exception:
                price_upper, std_upper, price_upper_w, midprice_w = [None]*4

            _comp_data = [filter]
            for fi in filters_to_add_as_columns:
                try:
                    _comp_data.append(filter[fi])
                except Exception:
                    _comp_data.append(None)
            _comp_data += [price, std, price_upper, std_upper,
                           delta, delta_std, gamma, gamma_std,
                           theta, theta_std, rho, rho_std, vega, vega_std,
                           price_w, price_upper_w, midprice_w,
                           delta_w, gamma_w, theta_w, rho_w, vega_w,
                           comp_time, time_w]
            comp_data.append(_comp_data)

    cols = ["desc"] + filters_to_add_as_columns + \
           ["price", "std", "price_upperbound", "std_priceupperbound",
            "delta", "delta_std", "gamma", "gamma_std",
            "theta", "theta_std", "rho", "rho_std", "vega", "vega_std",
            "price_w", "price_upperbound_w", "midprice_w",
            "delta_w", "gamma_w", "theta_w", "rho_w", "vega_w",
            "comp_time_median", "comp_time_median_w"]
    comp_df = pd.DataFrame(
        data=comp_data,
        columns=cols)
    if sort:
        comp_df = comp_df.sort_values(by=sortby, ascending=sort_ascending)

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
        save_path="../latex/plots/", filters_to_add_as_columns=[]):
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
    df3 = pd.merge(df1, df2, on=filters_to_add_as_columns, how="outer")
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


def combine_with_dos_table(
        save_path="../latex/plots/",
        filters_to_add_as_columns=["nb_stocks", "spot"]):
    save_comp_file = "{}comparison.csv".format(save_path)
    df = pd.read_csv(save_comp_file)
    df["time_w"] = df["comp_time_median"].apply(lambda s: "{:.0f}s".format(s))
    cols = ["price_w", "midprice_w", "price_upperbound_w", "time_w"]
    df = df[filters_to_add_as_columns+cols]
    df2 = pd.read_csv(io.StringIO(dos_table), index_col=None, sep=" ")
    df2["dos_time_tot"] = df2["timel"] + df2["timeu"]
    df2["dos_mid_w"] = df2["dos_mid"].apply(lambda s: "{:.3f}".format(s))
    df2["dos_lower_w"] = df2["dos_lower"].apply(lambda s: "{:.3f}".format(s))
    df2["dos_upper_w"] = df2["dos_upper"].apply(lambda s: "{:.3f}".format(s))
    df2["dos_time_tot_w"] = df2["dos_time_tot"].apply(
        lambda s: "{:.0f}s".format(s))
    df2 = df2[filters_to_add_as_columns+
              ["dos_lower_w", "dos_upper_w", "dos_mid_w", "dos_time_tot_w"]]
    df3 = pd.merge(df, df2, on=filters_to_add_as_columns, how="left")
    df3.rename(columns=lambda s: s.replace("_", ""), inplace=True)
    save_comp_file = "{}upper_bound_comparison_with_dos.csv".format(save_path)
    df3.to_csv(save_comp_file, index=False)




if __name__ == '__main__':
    filter_greeks = []
    for strike in [36, 40, 44]:
        filter_greeks += [{"algo": "B",
                           "strike": strike,
                           "greeks_method": "central",
                           "nb_dates": 50000,
                           "model": "BlackScholes",},]
        for method in ["central", "regression"]:
            for algo in ["LSM", "RLSMSoftplus", "FQI", "RFQI", "NLSM",]:
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



    filter_upper_bounds = []
    for spot in [90, 100, 110]:
        for nb_stocks in [2, 3, 5, 10, 20, 30, 50, 100, 200, 500]:
            filter_upper_bounds += [
                {"algo": "RLSMSoftplus",
                 "strike": 100,
                 "spot": spot,
                 "nb_stocks": nb_stocks,
                 "nb_paths": 100000,
                 "hidden_size": 100,
                 "nb_dates": 9,
                 "maturity": 3,
                 "model": "BlackScholes", }, ]

    filters_to_add_as_columns = ["algo", "nb_stocks", "spot"]
    get_comparison_csv(
        filter_upper_bounds, sortby=["nb_stocks", "spot"],
        sort_ascending=True,
        filters_to_add_as_columns=filters_to_add_as_columns)
    combine_with_dos_table()


    filter_sens_rand = []
    for strike in [100]:
        for algo in ["RLSMSoftplus", "RLSMSoftplusReinit", "NLSM"]:
            for hidden_size in [20, ]:
                for epochs in [10, 30, 50, 100]:
                    filter_sens_rand += [
                        {"algo": algo,
                         "strike": strike,
                         "hidden_size": hidden_size,
                         "nb_epochs": epochs,
                         "model": "BlackScholes", },
                    ]

    filters_to_add_as_columns = ["algo", "strike", "greeks_method",
                                 "hidden_size", "nb_epochs"]
    get_comparison_csv(
        filter_sens_rand, sort=False,
        filters_to_add_as_columns=filters_to_add_as_columns)




