import numpy as np
import pandas as pd


def main(args):
    best_file = None

    best_val = -np.inf

    for emitter_file, spot_file in zip(args.emitter_files, args.spot_files):
        df_emitter = pd.read_csv(emitter_file, sep="\t")

        if df_emitter.empty:
            continue

        precision = df_emitter["exc_precision"]

        recall = df_emitter["exc_recall"]

        val = compute_average_metric(precision, recall)

        if val > best_val:
            best_file = spot_file

            best_val = val

    df = pd.read_csv(best_file, sep="\t")

    df["decoder"] = "jsit"

    df.to_csv(args.out_file, index=False, sep="\t")


def compute_average_metric(precision, recall):
    return float(max(0.0, -np.sum(np.diff(recall) * np.array(precision)[:-1])))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--emitter-files", nargs="+", required=True)

    parser.add_argument("-s", "--spot-files", nargs="+", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    cli_args = parser.parse_args()

    main(cli_args)
