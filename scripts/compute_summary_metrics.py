import numpy as np
import pandas as pd


def main(args):
    metrics = {}

    df_bulk = pd.read_csv(args.bulk_file, sep="\t")

    metrics["run"] = df_bulk.iloc[0]["run"]

    metrics["replicate"] = df_bulk.iloc[0]["replicate"]

    metrics["decoder"] = df_bulk.iloc[0]["decoder"]

    metrics["score"] = df_bulk.iloc[0]["score"]

    df_bulk["norm_emitters"] = df_bulk["num_emitters"] / df_bulk["num_emitters"].max()

    metrics["average_r"] = compute_average_metric(df_bulk["r"], df_bulk["norm_emitters"])

    metrics["average_rho"] = compute_average_metric(df_bulk["rho"], df_bulk["norm_emitters"])

    metrics["max_r"] = df_bulk["r"].max()

    metrics["num_emitters_max_r"] = df_bulk[df_bulk["r"] == df_bulk["r"].max()]["num_emitters"].max()

    metrics["max_rho"] = df_bulk["rho"].max()

    metrics["num_emitters_max_rho"] = df_bulk[df_bulk["rho"] == df_bulk["rho"].max()]["num_emitters"].max()

    df_emitter = pd.read_csv(args.emitter_file, sep="\t")
    for prefix in ["exc", "loc"]:
        precision = df_emitter[f"{prefix}_precision"]

        recall = df_emitter[f"{prefix}_recall"]

        df_emitter[f"{prefix}_f1"] = 2 * (precision * recall) / (precision + recall)

        metrics[f"{prefix}_average_precision"] = compute_average_metric(precision, recall)

        metrics[f"{prefix}_max_f1"] = df_emitter[f"{prefix}_f1"].max()

    metrics = pd.DataFrame([metrics])

    metrics.to_csv(args.out_file, index=False, sep="\t")


def compute_average_metric(precision, recall):
    return float(max(0.0, -np.sum(np.diff(recall) * np.array(precision)[:-1])))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--bulk-file", required=True)

    parser.add_argument("-e", "--emitter-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    cli_args = parser.parse_args()

    main(cli_args)
