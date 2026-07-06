from scipy.stats import pearsonr, spearmanrho

import numpy as np
import pandas as pd


def main(args):
    df_pred = pd.read_csv(args.pred_file, sep="\t")

    df_true = pd.read_csv(args.true_file, sep="\t")

    if args.decoder == "bardensr":
        scores = ["evidence"]

    elif args.decoder == "deepcell":
        scores = ["probability"]

    elif args.decoder == "jsit":
        scores = ["mean_intensity"]

    elif args.decoder == "savannah":
        scores = ["dist_min", "dist_mean", "prob_max", "prob_mean"]

    else:
        scores = ["mean_distance", "min_distance", "mean_intensity", "max_intensity"]

    out_df = []

    for s in scores:
        if s in ["evidence", "max_intensity", "mean_intensity", "probability", "prob_max", "prob_mean"]:
            score_is_pos = True

        else:
            score_is_pos = False

        out_df.append(get_score_df(df_pred, df_true, s, score_is_pos=score_is_pos))

    out_df = pd.concat(out_df)

    out_df.insert(0, "run", args.run)

    out_df.insert(1, "replicate", args.replicate)

    out_df.insert(2, "decoder", args.decoder)

    out_df.to_csv(args.out_file, index=False, sep="\t")


def get_score_df(df_pred, df_true, score, score_is_pos=False):
    if df_pred.empty:
        return pd.DataFrame(
            [{"score": score, "threshold": 0, "r": 0, "r_p": 1, "rho": 0, "rho_p": 1, "num_emitters": 0}]
        )

    # Handle whether higher or lower is better for the score key
    if score_is_pos:
        df_pred["score"] = -df_pred[score]

    else:
        df_pred["score"] = df_pred[score]

    out_df = []

    for t in sorted(df_pred["score"].unique(), reverse=True):
        row = {"threshold": t}

        row.update(compute_correlation_stats(df_pred[df_pred["score"] <= t], df_true))

        out_df.append(row)

    out_df = pd.DataFrame(out_df)

    if score_is_pos:
        out_df["threshold"] = -out_df["threshold"]

    out_df.insert(0, "score", score)

    return out_df


def compute_correlation_stats(df_pred, df_true):
    counts_pred = np.log1p(df_pred["target"].value_counts())

    counts_true = np.log1p(df_true["target"].value_counts())

    counts = pd.concat([counts_pred, counts_true], axis=1).fillna(0)

    counts.columns = "pred", "true"

    pearson_result = pearsonr(counts["pred"], counts["true"])

    spearman_result = spearmanrho(counts["pred"], counts["true"])

    return {
        "r": pearson_result.statistic,
        "r_p": pearson_result.pvalue,
        "rho": spearman_result.statistic,
        "rho_p": spearman_result.pvalue,
        "num_emitters": df_pred.shape[0],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pred-file", required=True)

    parser.add_argument("-t", "--true-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("--decoder", required=True)

    parser.add_argument("--replicate", required=True)

    parser.add_argument("--run", required=True)

    cli_args = parser.parse_args()

    main(cli_args)
