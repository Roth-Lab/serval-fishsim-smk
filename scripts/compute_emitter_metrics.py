from scipy.spatial import cKDTree

import numpy as np
import pandas as pd


def main(args):
    df_pred = pd.read_csv(args.pred_file, sep="\t")

    df_pred = df_pred.rename(columns={"x": "y", "y": "x"})

    # Handle whether higher or lower is better for the score key
    if "intensity" in args.score:
        df_pred["score"] = -df_pred[args.score]

    else:
        df_pred["score"] = df_pred[args.score]

    df_true = pd.read_csv(args.true_file, sep="\t")

    pred_score_df = get_pred_score_df(df_pred, df_true, nn_dist=args.nn_dist)

    true_score_df = get_true_score_df(df_pred, df_true, nn_dist=args.nn_dist)

    exc_metrics_df = get_metrics(pred_score_df, true_score_df, prefix="exc")

    loc_metrics_df = get_metrics(pred_score_df, true_score_df, prefix="loc")

    out_df = pd.concat([exc_metrics_df, loc_metrics_df], axis=1)

    out_df = out_df.reset_index()

    if "intensity" in args.score:
        out_df["threshold"] = -out_df["threshold"]

    out_df.insert(0, "run", args.run)

    out_df.insert(1, "replicate", args.replicate)

    out_df.insert(2, "decoder", args.decoder)

    out_df.to_csv(args.out_file, index=False, sep="\t")


def get_pred_score_df(df_pred, df_true, nn_dist=1):
    """
    Compute matches from predicted spots to true spots.
    """
    score_df = []

    tree = cKDTree(df_true[["x", "y"]].values)

    idxs = tree.query_ball_point(df_pred[["x", "y"]].values, nn_dist)

    for i, nn_i in enumerate(idxs):
        loc_match = False

        exc_match = False

        score = df_pred.iloc[i]["score"]

        for j in nn_i:
            loc_match = True

            if df_pred.iloc[i]["target"] == df_true.iloc[j]["target"]:
                exc_match = True

        row = {"loc_match": loc_match, "exc_match": exc_match, "score": score}

        score_df.append(row)

    return pd.DataFrame(score_df)


def get_true_score_df(df_pred, df_true, nn_dist=1):
    """
    Compute matches from true spots to predicted spots.

    If more than one predicted spot matches the true spot keep the one with the lowest score.
    """
    score_df = []

    tree = cKDTree(df_pred[["x", "y"]].values)

    idxs = tree.query_ball_point(df_true[["x", "y"]].values, nn_dist)

    for i, nn_i in enumerate(idxs):
        loc_match = False

        exc_match = False

        loc_score = np.inf

        exc_score = np.inf

        for j in nn_i:
            loc_match = True

            if df_pred.iloc[j]["score"] < loc_score:
                loc_score = df_pred.iloc[j]["score"]

            if df_true.iloc[i]["target"] == df_pred.iloc[j]["target"]:
                exc_match = True

                if df_pred.iloc[j]["score"] < exc_score:
                    exc_score = df_pred.iloc[j]["score"]

        row = {"loc_match": loc_match, "exc_match": exc_match, "loc_score": loc_score, "exc_score": exc_score}

        score_df.append(row)

    return pd.DataFrame(score_df)


def get_metrics(pred_score_df, true_score_df, prefix="exc"):
    metrics_df = []

    for t in sorted(pred_score_df["score"].unique(), reverse=True):
        df_t = pred_score_df[pred_score_df["score"] <= t]

        tp = df_t[f"{prefix}_match"].sum()

        fp = df_t.shape[0] - tp

        df_t = true_score_df[true_score_df[f"{prefix}_score"] <= t]

        fn = true_score_df.shape[0] - df_t[f"{prefix}_match"].sum()

        row = {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp), "recall": tp / (tp + fn)}

        metrics_df.append(row)

    metrics_df = pd.DataFrame(metrics_df)

    metrics_df = metrics_df.set_index("threshold")

    metrics_df = metrics_df.rename(columns=lambda x: f"{prefix}_{x}")

    return metrics_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pred-file", required=True)

    parser.add_argument("-t", "--true-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("--decoder", required=True)

    parser.add_argument("--replicate", required=True)

    parser.add_argument("--run", required=True)

    parser.add_argument("--nn-dist", default=1, type=float)

    parser.add_argument(
        "--score",
        default="mean_distance",
        choices=["mean_distance", "min_distance", "mean_intensity", "max_intensity"],
    )

    cli_args = parser.parse_args()

    main(cli_args)
