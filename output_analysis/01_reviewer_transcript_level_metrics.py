import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import auc

OUTDIR = r"C:\Users\jenkints\Documents\GitHub\serval-fishsim-smk\output\paper_new\reviewer_transcript_level_metrics"
os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------------------
# 1. Load data / clean decoder names
# --------------------------------------------------
base_dir = Path(r"C:\Users\jenkints\Documents\GitHub\serval-fishsim-smk\output\paper_new")

emitter_df = pd.read_csv(base_dir / "emitter_metrics.tsv.gz", sep="\t")

# Remove old cosine decoder
emitter_df = emitter_df[emitter_df["decoder"] != "cosine"].copy()

print("Decoders included:")
print(sorted(emitter_df["decoder"].unique()))

# --------------------------------------------------
# 2. Safe division helper
# --------------------------------------------------
def safe_divide(num, den):
    return np.where(den > 0, num / den, np.nan)

# --------------------------------------------------
# 3. Recompute precision, recall, FDR, F1 from TP/FP/FN
# --------------------------------------------------
for mode in ["exc", "loc"]:

    tp = emitter_df[f"{mode}_tp"]
    fp = emitter_df[f"{mode}_fp"]
    fn = emitter_df[f"{mode}_fn"]

    emitter_df[f"{mode}_precision_calc"] = safe_divide(tp, tp + fp)
    emitter_df[f"{mode}_recall_calc"] = safe_divide(tp, tp + fn)
    emitter_df[f"{mode}_fdr_calc"] = safe_divide(fp, tp + fp)

    emitter_df[f"{mode}_f1_calc"] = safe_divide(
        2
        * emitter_df[f"{mode}_precision_calc"]
        * emitter_df[f"{mode}_recall_calc"],
        emitter_df[f"{mode}_precision_calc"]
        + emitter_df[f"{mode}_recall_calc"],
    )

# --------------------------------------------------
# 4. Compute average precision from threshold sweep
# --------------------------------------------------
group_cols = ["run", "replicate", "decoder"]

ap_rows = []

for (run, replicate, decoder), g in emitter_df.groupby(group_cols):

    row = {
        "run": run,
        "replicate": replicate,
        "decoder": decoder,
    }

    for mode in ["exc", "loc"]:

        precision_col = f"{mode}_precision_calc"
        recall_col = f"{mode}_recall_calc"

        pr = g[[precision_col, recall_col]].dropna().copy()

        if pr.empty:
            row[f"{mode}_average_precision"] = np.nan
            continue

        # Sort by recall increasing for PR integration
        pr = pr.sort_values(recall_col)

        # Remove duplicate recall values, keeping highest precision
        pr = (
            pr.groupby(recall_col, as_index=False)[precision_col]
            .max()
            .sort_values(recall_col)
        )

        recall = pr[recall_col].to_numpy()
        precision = pr[precision_col].to_numpy()

        # Add endpoints if needed
        if len(recall) == 0:
            ap = np.nan
        else:
            if recall[0] > 0:
                recall = np.insert(recall, 0, 0.0)
                precision = np.insert(precision, 0, precision[0])

            if recall[-1] < 1:
                recall = np.append(recall, 1.0)
                precision = np.append(precision, precision[-1])

            ap = auc(recall, precision)

        row[f"{mode}_average_precision"] = ap

    ap_rows.append(row)

ap_df = pd.DataFrame(ap_rows)

# --------------------------------------------------
# 5. For each run/replicate/decoder, select threshold with max F1
# --------------------------------------------------
summary_rows = []

for mode in ["exc", "loc"]:

    f1_col = f"{mode}_f1_calc"

    idx = (
        emitter_df
        .dropna(subset=[f1_col])
        .groupby(group_cols)[f1_col]
        .idxmax()
    )

    best = emitter_df.loc[idx].copy()

    best_summary = best[
        group_cols
        + [
            "score",
            "threshold",
            f"{mode}_tp",
            f"{mode}_fp",
            f"{mode}_fn",
            f"{mode}_precision_calc",
            f"{mode}_recall_calc",
            f"{mode}_fdr_calc",
            f"{mode}_f1_calc",
        ]
    ].copy()

    best_summary = best_summary.rename(columns={
        "score": f"{mode}_score_at_max_f1",
        "threshold": f"{mode}_threshold_at_max_f1",
        f"{mode}_tp": f"{mode}_tp_at_max_f1",
        f"{mode}_fp": f"{mode}_fp_at_max_f1",
        f"{mode}_fn": f"{mode}_fn_at_max_f1",
        f"{mode}_precision_calc": f"{mode}_precision_at_max_f1",
        f"{mode}_recall_calc": f"{mode}_recall_at_max_f1",
        f"{mode}_fdr_calc": f"{mode}_fdr_at_max_f1",
        f"{mode}_f1_calc": f"{mode}_max_f1",
    })

    summary_rows.append(best_summary)

reviewer_summary_df = summary_rows[0].merge(
    summary_rows[1],
    on=group_cols,
    how="outer",
)

# Add AP columns
reviewer_summary_df = reviewer_summary_df.merge(
    ap_df,
    on=group_cols,
    how="left",
)

reviewer_summary_df.to_csv(
    os.path.join(OUTDIR, "reviewer_transcript_level_summary_by_replicate.csv"),
    index=False,
)

# --------------------------------------------------
# 6. Aggregate by run and decoder
# --------------------------------------------------
metric_cols = [
    "exc_precision_at_max_f1",
    "exc_recall_at_max_f1",
    "exc_fdr_at_max_f1",
    "exc_max_f1",
    "exc_average_precision",
    "loc_precision_at_max_f1",
    "loc_recall_at_max_f1",
    "loc_fdr_at_max_f1",
    "loc_max_f1",
    "loc_average_precision",
]

summary_by_run_decoder = (
    reviewer_summary_df
    .groupby(["run", "decoder"])[metric_cols]
    .agg(["mean", "median", "std", "sem"])
)

summary_by_run_decoder.columns = [
    "_".join(col).strip()
    for col in summary_by_run_decoder.columns.values
]

summary_by_run_decoder = summary_by_run_decoder.reset_index()

summary_by_run_decoder.to_csv(
    os.path.join(OUTDIR, "reviewer_transcript_level_summary_by_run_decoder.csv"),
    index=False,
)

# --------------------------------------------------
# 7. Overall aggregate by decoder
# --------------------------------------------------
summary_by_decoder = (
    reviewer_summary_df
    .groupby("decoder")[metric_cols]
    .agg(["mean", "median", "std", "sem"])
)

summary_by_decoder.columns = [
    "_".join(col).strip()
    for col in summary_by_decoder.columns.values
]

summary_by_decoder = summary_by_decoder.reset_index()

summary_by_decoder.to_csv(
    os.path.join(OUTDIR, "reviewer_transcript_level_summary_by_decoder.csv"),
    index=False,
)

# --------------------------------------------------
# 8. Long format for plotting
# --------------------------------------------------
plot_metrics = [
    "exc_precision_at_max_f1",
    "exc_recall_at_max_f1",
    "exc_fdr_at_max_f1",
    "exc_max_f1",
    "exc_average_precision",
    "loc_precision_at_max_f1",
    "loc_recall_at_max_f1",
    "loc_fdr_at_max_f1",
    "loc_max_f1",
    "loc_average_precision",
]

metric_labels = {
    "exc_precision_at_max_f1": "Exact precision",
    "exc_recall_at_max_f1": "Exact recall",
    "exc_fdr_at_max_f1": "Exact FDR",
    "exc_max_f1": "Exact F1",
    "exc_average_precision": "Exact average precision",
    "loc_precision_at_max_f1": "Localization precision",
    "loc_recall_at_max_f1": "Localization recall",
    "loc_fdr_at_max_f1": "Localization FDR",
    "loc_max_f1": "Localization F1",
    "loc_average_precision": "Localization average precision",
}

long_df = reviewer_summary_df.melt(
    id_vars=["run", "replicate", "decoder"],
    value_vars=plot_metrics,
    var_name="metric",
    value_name="value",
)

long_df["metric_label"] = long_df["metric"].map(metric_labels)

long_df.to_csv(
    os.path.join(OUTDIR, "reviewer_transcript_level_long.csv"),
    index=False,
)

# --------------------------------------------------
# 9. Point plots
# --------------------------------------------------
def plot_point_metric(metric):

    sub = long_df[long_df["metric"] == metric].copy()

    plt.figure(figsize=(12, 5))

    ax = sns.pointplot(
        data=sub,
        x="run",
        y="value",
        hue="decoder",
        errorbar="se",
        dodge=0.3,
        markers="o",
        linestyles="-",
    )

    ax.set_title(metric_labels[metric])
    ax.set_xlabel("Simulation scenario")
    ax.set_ylabel(metric_labels[metric])
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)

    ax.legend(
        title="Decoder",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    plt.tight_layout()

    fname = f"pointplot_{metric}.png"
    plt.savefig(os.path.join(OUTDIR, fname), dpi=300)
    plt.close()

    print(f"Saved {fname}")


for metric in plot_metrics:
    plot_point_metric(metric)

# --------------------------------------------------
# 10. Scenario win counts
# --------------------------------------------------
winner_rows = []

for metric in plot_metrics:

    tmp = (
        reviewer_summary_df
        .groupby(["run", "decoder"])[metric]
        .mean()
        .reset_index()
    )

    if "fdr" in metric:
        idx = tmp.groupby("run")[metric].idxmin()
    else:
        idx = tmp.groupby("run")[metric].idxmax()

    winners = tmp.loc[idx, ["run", "decoder", metric]].copy()
    winners["metric"] = metric
    winners = winners.rename(columns={metric: "winning_value"})

    winner_rows.append(winners)

winner_df = pd.concat(winner_rows, ignore_index=True)

winner_counts = (
    winner_df
    .groupby(["metric", "decoder"])
    .size()
    .reset_index(name="num_scenarios_won")
    .sort_values(["metric", "num_scenarios_won"], ascending=[True, False])
)

winner_df.to_csv(
    os.path.join(OUTDIR, "scenario_winners_by_metric.csv"),
    index=False,
)

winner_counts.to_csv(
    os.path.join(OUTDIR, "scenario_winner_counts_by_metric.csv"),
    index=False,
)

print("Saved reviewer-facing transcript-level metrics to:", OUTDIR)
print("\nOverall summary by decoder:")
print(summary_by_decoder)

print("\nScenario winner counts:")
print(winner_counts)