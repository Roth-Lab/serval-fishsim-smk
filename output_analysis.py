from pathlib import Path
import pandas as pd

base_dir = Path(r"C:\Users\jenkints\Documents\GitHub\serval-fishsim-smk\output\paper_new")

bulk_df = pd.read_csv(base_dir / "bulk_metrics.tsv.gz", sep="\t")
emitter_df = pd.read_csv(base_dir / "emitter_metrics.tsv.gz", sep="\t")
summary_df = pd.read_csv(base_dir / "summary_metrics.tsv.gz", sep="\t")

# Remove old cosine decoder
summary_df = summary_df[summary_df["decoder"] != "cosine"].copy()

print("Bulk metrics:")
print(bulk_df.head())
print()

print("Emitter metrics:")
print(emitter_df.head())
print()

print("Summary metrics:")
print(summary_df.head())
print()

print("Shapes:")
print(f"bulk_df:    {bulk_df.shape}")
print(f"emitter_df: {emitter_df.shape}")
print(f"summary_df: {summary_df.shape}")




import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTDIR = r"C:\Users\jenkints\Documents\GitHub\serval-fishsim-smk\output\paper_new\metric_investigation_plots"
os.makedirs(OUTDIR, exist_ok=True)

# Exclude old cosine; cosine-np is the actual Cosine method
summary_df = summary_df[summary_df["decoder"] != "cosine"].copy()

print("Decoders included:")
print(sorted(summary_df["decoder"].unique()))

# -----------------------------
# Metric groups
# -----------------------------
transcript_metrics = [
    "exc_average_precision",
    "exc_max_f1",
    "loc_average_precision",
    "loc_max_f1",
]

abundance_metrics = [
    "max_r",
    "max_rho",
]

emitter_metrics = [
    "num_emitters_max_r",
    "num_emitters_max_rho",
]

all_metrics = transcript_metrics + abundance_metrics + emitter_metrics
all_metrics = [m for m in all_metrics if m in summary_df.columns]

metric_labels = {
    "exc_average_precision": "Exact-match AP",
    "exc_max_f1": "Exact-match max F1",
    "loc_average_precision": "Localization AP",
    "loc_max_f1": "Localization max F1",
    "max_r": "Max Pearson r",
    "max_rho": "Max Spearman ρ",
    "num_emitters_max_r": "Emitters at max Pearson r",
    "num_emitters_max_rho": "Emitters at max Spearman ρ",
}

# -----------------------------
# Long-format dataframe
# -----------------------------
long_df = summary_df.melt(
    id_vars=["run", "replicate", "decoder"],
    value_vars=all_metrics,
    var_name="metric",
    value_name="value",
)

long_df["metric_label"] = long_df["metric"].map(metric_labels)

# -----------------------------
# Summary tables
# -----------------------------
summary_table = (
    long_df
    .groupby(["metric", "run", "decoder"])["value"]
    .agg(["mean", "median", "std", "sem"])
    .reset_index()
)

summary_table.to_csv(
    os.path.join(OUTDIR, "metric_summary_by_run_decoder.csv"),
    index=False,
)

overall_summary = (
    long_df
    .groupby(["metric", "decoder"])["value"]
    .agg(["mean", "median", "std", "sem"])
    .reset_index()
)

overall_summary.to_csv(
    os.path.join(OUTDIR, "metric_summary_overall_decoder.csv"),
    index=False,
)

# -----------------------------
# Boxplots
# -----------------------------
def plot_metric_boxplot(df, metric, outdir=OUTDIR):
    sub = df[df["metric"] == metric].copy()

    if sub.empty:
        print(f"Skipping {metric}: no data")
        return

    plt.figure(figsize=(14, 5))

    ax = sns.boxplot(
        data=sub,
        x="run",
        y="value",
        hue="decoder",
        showfliers=False,
    )

    sns.stripplot(
        data=sub,
        x="run",
        y="value",
        hue="decoder",
        dodge=True,
        alpha=0.25,
        size=2,
        linewidth=0,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    n_decoders = sub["decoder"].nunique()

    ax.legend(
        handles[:n_decoders],
        labels[:n_decoders],
        title="Decoder",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    ax.set_title(metric_labels.get(metric, metric))
    ax.set_xlabel("Simulation scenario")
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.tick_params(axis="x", rotation=45)

    if metric in transcript_metrics + abundance_metrics:
        ax.set_ylim(0, 1.05)

    plt.tight_layout()

    fname = f"boxplot_{metric}.png"
    plt.savefig(os.path.join(outdir, fname), dpi=300)
    plt.close()

    print(f"Saved {fname}")


for metric in all_metrics:
    plot_metric_boxplot(long_df, metric)

# -----------------------------
# Mean ± SEM line plots
# -----------------------------
def plot_metric_lineplot(df, metric, outdir=OUTDIR):
    sub = df[df["metric"] == metric].copy()

    if sub.empty:
        print(f"Skipping {metric}: no data")
        return

    plt.figure(figsize=(12, 5))

    ax = sns.pointplot(
        data=sub,
        x="run",
        y="value",
        hue="decoder",
        errorbar="se",
        dodge=0.25,
        markers="o",
        linestyles="-",
    )

    ax.set_title(metric_labels.get(metric, metric))
    ax.set_xlabel("Simulation scenario")
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.tick_params(axis="x", rotation=45)

    if metric in transcript_metrics + abundance_metrics:
        ax.set_ylim(0, 1.05)

    ax.legend(
        title="Decoder",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    plt.tight_layout()

    fname = f"mean_sem_{metric}.png"
    plt.savefig(os.path.join(outdir, fname), dpi=300)
    plt.close()

    print(f"Saved {fname}")


for metric in all_metrics:
    plot_metric_lineplot(long_df, metric)

# -----------------------------
# Average-rank table
# Higher is better for AP, F1, Pearson, Spearman.
# Do not include num_emitters in ranking.
# -----------------------------
rank_metrics = transcript_metrics + abundance_metrics
rank_metrics = [m for m in rank_metrics if m in summary_df.columns]

rank_rows = []

for metric in rank_metrics:
    tmp = summary_df[["run", "replicate", "decoder", metric]].dropna().copy()

    tmp["rank"] = tmp.groupby(["run", "replicate"])[metric].rank(
        ascending=False,
        method="average",
    )

    rank_summary = (
        tmp.groupby("decoder")["rank"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    rank_summary["metric"] = metric
    rank_rows.append(rank_summary)

rank_df = pd.concat(rank_rows, ignore_index=True)
rank_df = rank_df[["metric", "decoder", "mean", "median", "std"]]

rank_df.to_csv(
    os.path.join(OUTDIR, "average_rank_by_metric.csv"),
    index=False,
)

print("Saved plots and tables to:", OUTDIR)
print(rank_df)