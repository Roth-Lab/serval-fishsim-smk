from scipy.stats import pearsonr, spearmanrho

import pandas as pd


def main(args):
    df_pred = pd.read_csv(args.pred_file, sep="\t")

    # Handle whether higher or lower is better for the score key
    if "intensity" in args.score:
        df_pred["score"] = -df_pred[args.score]

    else:
        df_pred["score"] = df_pred[args.score]

    df_true = pd.read_csv(args.true_file, sep="\t")

    out_df = []

    for t in sorted(df_pred["score"].unique(), reverse=True):
        row = {"threshold": t}

        row.update(compute_correlation_stats(df_pred[df_pred["score"] <= t], df_true))

        out_df.append(row)

    out_df = pd.DataFrame(out_df)

    if "intensity" in args.score:
        out_df["threshold"] = -out_df["threshold"]

    out_df.insert(0, "run", args.run)

    out_df.insert(1, "replicate", args.replicate)

    out_df.insert(2, "decoder", args.decoder)

    out_df.insert(3, "score", args.score)

    out_df.to_csv(args.out_file, index=False, sep="\t")


def compute_correlation_stats(df_pred, df_true):
    counts_true = df_true["target"].value_counts()

    counts_pred = df_pred["target"].value_counts()

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

    parser.add_argument(
        "--score",
        default="mean_distance",
        choices=["mean_distance", "min_distance", "mean_intensity", "max_intensity"],
    )

    cli_args = parser.parse_args()

    main(cli_args)
