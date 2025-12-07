# -*- coding: utf-8 -*-
"""
LOB_GAN_synthetic_compare.py

Use the trained generator to produce synthetic LOB sequences for a few
testing days, compare them to the real order books, and save plots.

Usage:
    python3 LOB_GAN_synthetic.py --stock 0005
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# -----------------------------
#  Model + data utilities
# -----------------------------


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lay1 = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.lay2 = nn.Sequential(
            nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40)
        )
        self.lay3 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay4 = nn.Sequential(
            nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40)
        )
        self.lay5 = nn.GRU(
            40, 19, num_layers=1, batch_first=True
        )  # layer < 20 nodes to avoid repetition
        self.lay6 = nn.Sequential(
            nn.Linear(19, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40)
        )
        self.lay7 = nn.GRU(40, 20, num_layers=1, batch_first=True)
        self.lay8 = nn.Sequential(
            nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 20)
        )

    def forward(self, x):
        y, _ = self.lay1(x)
        z = self.lay2(y)
        u, _ = self.lay3(z)
        v = self.lay4(u)
        w, _ = self.lay5(v)
        o = self.lay6(w)
        p, _ = self.lay7(o)
        # q = self.lay8(p)
        return p


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepareMinutelyData(df: pd.DataFrame, tradingDays: list) -> pd.DataFrame:
    """Same logic as in training/testing files."""
    if df.empty:
        return None

    df["bfValue"] = df["lastPx"] * df["size"]
    df["bfValue"] = df["bfValue"].ffill()
    df["cumValue"] = df.groupby("date")["bfValue"].cumsum()
    df = df[df["SP1"] > 0]
    df = df[df["BP1"] > 0]
    df = df[df["SP1"] - df["BP1"] > 0]

    for i in range(1, 6):
        df[f"SP{i}"] = df[f"SP{i}"] / 100
        df[f"BP{i}"] = df[f"BP{i}"] / 100
        df[f"SV{i}"] = df[f"SV{i}"] * 1000
        df[f"BV{i}"] = df[f"BV{i}"] * 1000

    df["lastPx"] = df["lastPx"] / 100
    df["size"] = df["size"] * 1000
    df["volume"] = df["volume"] * 1000
    df["lastPx"] = df.groupby("date")["lastPx"].ffill()
    df["size"] = df.groupby("date")["size"].transform(lambda x: x.fillna(0))

    df["value"] = df.groupby("date")["cumValue"].diff()
    df["value"] = df["value"].fillna(df["bfValue"])
    df.drop(columns=["bfValue", "cumValue", "value"], inplace=True)

    # build datetime index
    df_DateTime = pd.to_datetime(
        df.date.astype(str) + " " + df.time.astype(str), format="%Y-%m-%d %H%M%S%f"
    )
    df["dt_index"] = df_DateTime
    df = df[~df.dt_index.duplicated(keep="last")]

    # bin to minutely
    binSize = "1min"
    df_minutely = df.groupby(
        pd.Grouper(key="dt_index", freq=binSize, closed="right", label="right")
    ).last()

    for i in range(1, 6):
        df_minutely.loc[:, f"SP{i}"] = df.groupby(
            pd.Grouper(key="dt_index", freq=binSize, closed="right", label="right")
        )[f"SP{i}"].last()
        df_minutely.loc[:, f"BP{i}"] = df.groupby(
            pd.Grouper(key="dt_index", freq=binSize, closed="right", label="right")
        )[f"BP{i}"].last()
        df_minutely.loc[:, f"SV{i}"] = df.groupby(
            pd.Grouper(key="dt_index", freq=binSize, closed="right", label="right")
        )[f"SV{i}"].last()
        df_minutely.loc[:, f"BV{i}"] = df.groupby(
            pd.Grouper(key="dt_index", freq=binSize, closed="right", label="right")
        )[f"BV{i}"].last()

    # session filter
    df_minutely = df_minutely.between_time("09:00:00", "13:25:00", inclusive="right")
    df_minutely["date"] = df_minutely.index.date
    df_minutely["ttime"] = df_minutely.index.time
    df_minutely["time"].fillna(df_minutely["ttime"], inplace=True)
    df_minutely.drop(columns=["ttime"], inplace=True)

    df_minutely = df_minutely[df_minutely["date"].astype(str).isin(tradingDays)]

    return df_minutely


# -----------------------------
#  Main: generate & compare
# -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=str, required=True, help="stock symbol")
    args = parser.parse_args()

    stock = args.stock
    stockDataDir = "data/"

    # columns to load from md files
    cols = [
        "date",
        "time",
        "lastPx",
        "size",
        "volume",
        "SP1",
        "BP1",
        "SV1",
        "BV1",
        "SP2",
        "BP2",
        "SV2",
        "BV2",
        "SP3",
        "BP3",
        "SV3",
        "BV3",
        "SP4",
        "BP4",
        "SV4",
        "BV4",
        "SP5",
        "BP5",
        "SV5",
        "BV5",
    ]

    # IMPORTANT:
    # paste in the SAME tradingDays list you used in your training/testing scripts.
    # For brevity, I don't repeat the full huge list here.
    tradingDays = [
        "2023-10-02",
        "2023-10-03",
        "2023-10-04",
        "2023-10-05",
        "2023-10-06",
        "2023-10-11",
        "2023-10-12",
        "2023-10-13",
        "2023-10-16",
        "2023-10-17",
        "2023-10-18",
        "2023-10-19",
        "2023-10-20",
        "2023-10-23",
        "2023-10-24",
        "2023-10-25",
        "2023-10-26",
        "2023-10-27",
        "2023-10-30",
        "2023-10-31",
        "2023-11-01",
        "2023-11-02",
        "2023-11-03",
        "2023-11-06",
        "2023-11-07",
        "2023-11-08",
        "2023-11-09",
        "2023-11-10",
        "2023-11-13",
        "2023-11-14",
        "2023-11-15",
        "2023-11-16",
        "2023-11-17",
        "2023-11-20",
        "2023-11-21",
        "2023-11-22",
        "2023-11-23",
        "2023-11-24",
        "2023-11-27",
        "2023-11-28",
        "2023-11-29",
        "2023-11-30",
        "2023-12-01",
        "2023-12-04",
        "2023-12-05",
        "2023-12-06",
        "2023-12-07",
        "2023-12-08",
        "2023-12-11",
        "2023-12-12",
        "2023-12-13",
        "2023-12-14",
        "2023-12-15",
        "2023-12-18",
        "2023-12-19",
        "2023-12-20",
        "2023-12-21",
        "2023-12-22",
        "2023-12-25",
        "2023-12-26",
        "2023-12-27",
        "2023-12-28",
        "2023-12-29",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
        "2024-01-15",
        "2024-01-16",
        "2024-01-17",
        "2024-01-18",
        "2024-01-19",
        "2024-01-22",
        "2024-01-23",
        "2024-01-24",
        "2024-01-25",
        "2024-01-26",
        "2024-01-29",
        "2024-01-30",
        "2024-01-31",
        "2024-02-01",
        "2024-02-02",
        "2024-02-15",
        "2024-02-16",
        "2024-02-19",
        "2024-02-20",
        "2024-02-21",
        "2024-02-22",
        "2024-02-23",
        "2024-02-26",
        "2024-02-27",
        "2024-02-29",
        "2024-03-01",
        "2024-03-04",
        "2024-03-05",
        "2024-03-06",
        "2024-03-07",
        "2024-03-08",
        "2024-03-11",
        "2024-03-12",
        "2024-03-13",
        "2024-03-14",
        "2024-03-15",
        "2024-03-18",
        "2024-03-19",
        "2024-03-20",
        "2024-03-21",
        "2024-03-22",
        "2024-03-25",
        "2024-03-26",
        "2024-03-27",
        "2024-03-28",
        "2024-03-29",
        "2024-04-01",
        "2024-04-02",
        "2024-04-03",
        "2024-04-08",
        "2024-04-09",
        "2024-04-10",
        "2024-04-11",
        "2024-04-12",
        "2024-04-15",
        "2024-04-16",
        "2024-04-17",
        "2024-04-18",
        "2024-04-19",
        "2024-04-22",
        "2024-04-23",
        "2024-04-24",
        "2024-04-25",
        "2024-04-26",
        "2024-04-29",
        "2024-04-30",
        "2024-05-02",
        "2024-05-03",
        "2024-05-06",
        "2024-05-07",
        "2024-05-08",
        "2024-05-09",
        "2024-05-10",
        "2024-05-13",
        "2024-05-14",
        "2024-05-15",
        "2024-05-16",
        "2024-05-17",
        "2024-05-20",
        "2024-05-21",
        "2024-05-22",
        "2024-05-23",
        "2024-05-24",
        "2024-05-27",
        "2024-05-28",
        "2024-05-29",
        "2024-05-30",
        "2024-05-31",
        "2024-06-03",
        "2024-06-04",
        "2024-06-05",
        "2024-06-06",
        "2024-06-07",
        "2024-06-11",
        "2024-06-12",
        "2024-06-13",
        "2024-06-14",
        "2024-06-17",
        "2024-06-18",
        "2024-06-19",
        "2024-06-20",
        "2024-06-21",
        "2024-06-24",
        "2024-06-25",
        "2024-06-26",
        "2024-06-27",
        "2024-06-28",
        "2024-07-01",
        "2024-07-02",
        "2024-07-03",
        "2024-07-04",
        "2024-07-05",
        "2024-07-08",
        "2024-07-09",
        "2024-07-10",
        "2024-07-11",
        "2024-07-12",
        "2024-07-15",
        "2024-07-16",
        "2024-07-17",
        "2024-07-18",
        "2024-07-19",
        "2024-07-22",
        "2024-07-23",
        "2024-07-26",
        "2024-07-29",
        "2024-07-30",
        "2024-07-31",
        "2024-08-01",
        "2024-08-02",
        "2024-08-05",
        "2024-08-06",
        "2024-08-07",
        "2024-08-08",
        "2024-08-09",
        "2024-08-12",
        "2024-08-13",
        "2024-08-14",
        "2024-08-15",
        "2024-08-16",
        "2024-08-19",
        "2024-08-20",
        "2024-08-21",
        "2024-08-22",
        "2024-08-23",
        "2024-08-26",
        "2024-08-27",
        "2024-08-28",
        "2024-08-29",
        "2024-08-30",
        "2024-09-02",
        "2024-09-03",
        "2024-09-04",
        "2024-09-05",
        "2024-09-06",
        "2024-09-09",
        "2024-09-10",
        "2024-09-11",
        "2024-09-12",
        "2024-09-13",
        "2024-09-16",
        "2024-09-18",
        "2024-09-19",
        "2024-09-20",
        "2024-09-23",
        "2024-09-24",
        "2024-09-25",
        "2024-09-26",
        "2024-09-27",
        "2024-09-30",
        "2024-10-01",
        "2024-10-02",
        "2024-10-03",
        "2024-10-04",
        "2024-10-07",
        "2024-10-08",
        "2024-10-09",
        "2024-10-11",
        "2024-10-14",
        "2024-10-15",
        "2024-10-16",
        "2024-10-17",
        "2024-10-18",
        "2024-10-21",
        "2024-10-22",
        "2024-10-23",
        "2024-10-24",
        "2024-10-25",
        "2024-10-28",
        "2024-10-29",
        "2024-10-30",
        "2024-10-31",
        "2024-11-01",
        "2024-11-04",
        "2024-11-05",
        "2024-11-06",
        "2024-11-07",
        "2024-11-08",
        "2024-11-11",
        "2024-11-12",
        "2024-11-13",
        "2024-11-14",
        "2024-11-15",
        "2024-11-18",
        "2024-11-19",
        "2024-11-20",
        "2024-11-21",
        "2024-11-22",
        "2024-11-25",
        "2024-11-26",
        "2024-11-27",
        "2024-11-28",
        "2024-11-29",
        "2024-12-02",
        "2024-12-03",
        "2024-12-04",
        "2024-12-05",
        "2024-12-06",
        "2024-12-09",
        "2024-12-10",
        "2024-12-11",
        "2024-12-12",
        "2024-12-13",
        "2024-12-16",
        "2024-12-17",
        "2024-12-18",
        "2024-12-19",
        "2024-12-20",
        "2024-12-23",
        "2024-12-24",
        "2024-12-25",
        "2024-12-26",
        "2024-12-27",
        "2024-12-30",
        "2024-12-31",
    ] # if you expose it there
    # Or, alternatively, manually define tradingDays = [...] here.

    print(f"Loading raw testing data for {stock}...")

    # testing months: Jan–Mar 2024 (same as your testing file)
    file1Path = os.path.join(stockDataDir, f"{stock}_md_202401_202401.csv.gz")
    file2Path = os.path.join(stockDataDir, f"{stock}_md_202402_202402.csv.gz")
    file3Path = os.path.join(stockDataDir, f"{stock}_md_202403_202403.csv.gz")

    df = pd.DataFrame()
    for path in [file1Path, file2Path, file3Path]:
        if os.path.exists(path):
            df = pd.concat([df, pd.read_csv(path, compression="gzip", usecols=cols)])
            print(f"Loaded {path}")
        else:
            print(f"Skipping missing snapshots: {path}")

    if df.empty:
        print(f"No md data for {stock}. Exiting.")
        return

    # minute-level data
    minutelyData = prepareMinutelyData(df, tradingDays)
    print("Minutely data generated.")

    # build daily sequences as in training/testing
    projdata = []
    columns = [
        "date",
        "time",
        "lastPx",
        "size",
        "volume",
        "SP5",
        "SP4",
        "SP3",
        "SP2",
        "SP1",
        "BP1",
        "BP2",
        "BP3",
        "BP4",
        "BP5",
        "SV5",
        "SV4",
        "SV3",
        "SV2",
        "SV1",
        "BV1",
        "BV2",
        "BV3",
        "BV4",
        "BV5",
    ]

    minutelyData = minutelyData.reset_index()  # dt_index becomes a column
    md = minutelyData.set_index("dt_index")

    seq_dates = []  # for reference (one per full day)

    for date, df_day in md.groupby("date"):
        if df_day.shape[0] == 265:
            projdata.append(df_day[columns].values)
            seq_dates.append(date)

    projdata = np.array(projdata)
    seq_dates = np.array(seq_dates)

    if projdata.shape[0] == 0:
        print("No days with exactly 265 minutes. Exiting.")
        return

    # normalization (same as training/testing)
    X = projdata[:, :, 5:].astype(float)  # take 20 LOB features

    X[:, :, -10:] = np.log(1 + X[:, :, -10:])
    X_mean = X.mean(axis=1)
    X_std = X.std(axis=1)

    X = np.transpose((np.transpose(X, (1, 0, 2)) - X_mean) / (2 * X_std), (1, 0, 2))
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # dataset & generator
    set_seed(307)
    dataset = MyDataset(torch.tensor(X, dtype=torch.float32))

    model_dir = f"data_{stock}"
    os.makedirs(model_dir, exist_ok=True)
    gen_path = os.path.join(model_dir, f"{stock}_generator1.pth")

    if not os.path.exists(gen_path):
        print(f"Generator file not found: {gen_path}")
        return

    generator = torch.load(gen_path, weights_only=False)
    generator.eval()
    print(f"Loaded generator from {gen_path}")

    # pick a few example days
    n_days = len(dataset)
    sample_indices = sorted(set([0, min(5, n_days - 1), min(10, n_days - 1)]))

    feature_cols = [
        "SP5",
        "SP4",
        "SP3",
        "SP2",
        "SP1",
        "BP1",
        "BP2",
        "BP3",
        "BP4",
        "BP5",
        "SV5",
        "SV4",
        "SV3",
        "SV2",
        "SV1",
        "BV1",
        "BV2",
        "BV3",
        "BV4",
        "BV5",
    ]

    print("Generating synthetic sequences and saving comparison plots...")

    with torch.no_grad():
        for idx in sample_indices:
            real_seq_norm = dataset[idx].unsqueeze(0)  # (1, 265, 20)
            gen_seq_norm = generator(real_seq_norm)    # (1, 265, 20)

            real_seq_norm = real_seq_norm.numpy()[0]   # (265, 20)
            gen_seq_norm = gen_seq_norm.numpy()[0]     # (265, 20)

            mu = X_mean[idx]       # (20,)
            sigma = X_std[idx]     # (20,)

            real_seq = real_seq_norm * (2 * sigma) + mu
            gen_seq = gen_seq_norm * (2 * sigma) + mu

            df_real = pd.DataFrame(real_seq, columns=feature_cols)
            df_gen = pd.DataFrame(gen_seq, columns=feature_cols)

            df_real["minute"] = np.arange(df_real.shape[0])
            df_gen["minute"] = np.arange(df_gen.shape[0])

            day_label = str(seq_dates[idx])

            # ---- Plot 1: best bid/ask intraday ----
            plt.figure(figsize=(8, 4))
            plt.plot(df_real["minute"], df_real["BP1"], label="Real BP1", alpha=0.8)
            plt.plot(df_real["minute"], df_real["SP1"], label="Real SP1", alpha=0.8)
            plt.plot(df_gen["minute"], df_gen["BP1"], "--", label="Synthetic BP1", alpha=0.8)
            plt.plot(df_gen["minute"], df_gen["SP1"], "--", label="Synthetic SP1", alpha=0.8)
            plt.title(f"{stock} – {day_label}: Best bid/ask over the day")
            plt.xlabel("Minute of day")
            plt.ylabel("Price")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f"{stock}_day{idx}_best_prices.png"), dpi=300)
            plt.close()

            # ---- Plot 2: spread intraday ----
            real_spread = df_real["SP1"] - df_real["BP1"]
            gen_spread = df_gen["SP1"] - df_gen["BP1"]

            plt.figure(figsize=(8, 4))
            plt.plot(df_real["minute"], real_spread, label="Real spread", alpha=0.8)
            plt.plot(df_gen["minute"], gen_spread, "--", label="Synthetic spread", alpha=0.8)
            plt.title(f"{stock} – {day_label}: Bid–ask spread over the day")
            plt.xlabel("Minute of day")
            plt.ylabel("Spread")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f"{stock}_day{idx}_spread.png"), dpi=300)
            plt.close()

            # ---- Plot 3: average depth profile ----
            real_bid_depth = df_real[["BV1", "BV2", "BV3", "BV4", "BV5"]].mean(axis=0)
            real_ask_depth = df_real[["SV1", "SV2", "SV3", "SV4", "SV5"]].mean(axis=0)
            gen_bid_depth = df_gen[["BV1", "BV2", "BV3", "BV4", "BV5"]].mean(axis=0)
            gen_ask_depth = df_gen[["SV1", "SV2", "SV3", "SV4", "SV5"]].mean(axis=0)

            levels = np.arange(1, 6)

            plt.figure(figsize=(6, 4))
            plt.plot(levels, real_bid_depth.values, "-o", label="Real bid depth")
            plt.plot(levels, gen_bid_depth.values, "--o", label="Synthetic bid depth")
            plt.plot(levels, real_ask_depth.values, "-s", label="Real ask depth")
            plt.plot(levels, gen_ask_depth.values, "--s", label="Synthetic ask depth")
            plt.xticks(levels)
            plt.xlabel("Level (1 = top of book)")
            plt.ylabel("Average depth")
            plt.title(f"{stock} – {day_label}: Average depth profile")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f"{stock}_day{idx}_depth_profile.png"), dpi=300)
            plt.close()

            print(f"Saved plots for day index {idx} ({day_label})")

    print("Done.")


if __name__ == "__main__":
    main()
