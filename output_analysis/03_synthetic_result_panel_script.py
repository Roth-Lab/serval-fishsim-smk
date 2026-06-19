import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Paths
# --------------------------------------------------
RESULTS_DIR = r"/projects/molonc/scratch/aroth/projects/serval/results/fishsim/paper_100"

ANALYSIS_DIR = "/projects/molonc/scratch/jtsui/serval-fishsim-smk/output/paper_100"

OUTDIR = r"{ANALYSIS_DIR}/figure2_synthetic_main"
os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
reviewer_summary_df = pd.read_csv(os.path.join(ANALYSIS_DIR, "reviewer_transcript_level_metrics", "reviewer_transcript_level_summary_by_replicate.csv"))

bulk_df = pd.read_csv(os.path.join(RESULTS_DIR, "bulk_metrics.tsv.gz"), sep="\t")

# --------------------------------------------------
# Clean decoders
# --------------------------------------------------
reviewer_summary_df = reviewer_summary_df[
    reviewer_summary_df["decoder"] != "cosine"
].copy()

bulk_df = bulk_df[
    bulk_df["decoder"] != "cosine"
].copy()

# --------------------------------------------------
# Compute max Pearson r from bulk_df
# --------------------------------------------------
idx = (
    bulk_df
    .dropna(subset=["r"])
    .groupby(["run", "replicate", "decoder"])["r"]
    .idxmax()
)

max_r_df = bulk_df.loc[idx, [
    "run", "replicate", "decoder", "r", "threshold", "num_emitters"
]].copy()

max_r_df = max_r_df.rename(columns={
    "r": "max_r",
    "threshold": "threshold_at_max_r",
    "num_emitters": "num_emitters_at_max_r",
})

# --------------------------------------------------
# Merge transcript-level metrics with max_r
# --------------------------------------------------
fig_df = reviewer_summary_df.merge(
    max_r_df,
    on=["run", "replicate", "decoder"],
    how="inner",
)

# --------------------------------------------------
# Metrics for Figure 2
# --------------------------------------------------
plot_metrics = [
    "exc_recall_at_max_f1",
    "exc_fdr_at_max_f1",
    "exc_max_f1",
    "max_r",
]

metric_labels = {
    "exc_recall_at_max_f1": "Exact recall",
    "exc_fdr_at_max_f1": "Exact FDR",
    "exc_max_f1": "Exact F1",
    "max_r": "Max Pearson r",
}

panel_titles = {
    "exc_recall_at_max_f1": "A. Transcript recall",
    "exc_fdr_at_max_f1": "B. False discovery rate",
    "exc_max_f1": "C. Transcript-level F1",
    "max_r": "D. Abundance recovery",
}

decoder_labels = {
    "cosine_np": "Cosine",
    "cosine-np": "Cosine",
    "scaled": "MERlin",
    "nn": "Nearest neighbor",
    "bardensr": "BarDensr",
    "deepcell": "DeepCell-Spots",
    "deepcell-spots": "DeepCell-Spots",
}

decoder_palette = {
    "BarDensr":         "#d62728",  # Red
    "Cosine":           "#2ca02c",  # Green
    "DeepCell-Spots":   "#9467bd",  # Purple
    "MERlin":           "#1f77b4",  # Blue
    "Nearest neighbor": "#ff7f0e",  # Orange
}

fig_df["decoder_label"] = fig_df["decoder"].map(decoder_labels).fillna(fig_df["decoder"])

# Optional: force scenario order if run names are scenario1, scenario2, ...
def scenario_sort_key(x):
    digits = "".join([c for c in str(x) if c.isdigit()])
    return int(digits) if digits else 999

run_order = sorted(fig_df["run"].unique(), key=scenario_sort_key)

# remove scenario11 from displayed order
run_order = [r for r in run_order if scenario_sort_key(r) != 11]

# --------------------------------------------------
# Make 2x2 Figure 2
# --------------------------------------------------
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")



fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 9),
    sharex=False
)

axes = axes.flatten()

legend_handles = None
legend_labels = None

for i, (ax, metric) in enumerate(zip(axes, plot_metrics)):

    sub = fig_df[
        ["run", "replicate", "decoder_label", metric]
    ].dropna().copy()

    sns.pointplot(
        data=sub,
        x="run",
        y=metric,
        hue="decoder_label",
        order=run_order,
        palette=decoder_palette,
        errorbar="se",
        dodge=0.15,
        markers="o",
        linestyles="-",
        ax=ax,
    )

    ax.set_title(
        panel_titles[metric],
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Simulation scenario")
    ax.set_ylabel(metric_labels[metric])

    # Save legend from first panel
    if i == 0:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

    # Remove subplot legends
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # Slanted scenario labels
    ax.tick_params(axis="x", rotation=45)

    # Consistent y-axis scaling
    if metric != "exc_fdr_at_max_f1":
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylim(
            0,
            max(
                0.05,
                sub[metric].max() * 1.15
            )
        )

# Single legend on right side
fig.legend(
    legend_handles,
    legend_labels,
    title="Decoder",
    loc="center left",
    bbox_to_anchor=(0.90, 0.5),
    frameon=True,
)

plt.subplots_adjust(
    left=0.08,
    right=0.85,
    bottom=0.12,
    top=0.92,
    wspace=0.28,
    hspace=0.45,
)

fig.savefig(
    os.path.join(
        OUTDIR,
        "Figure2_synthetic_main.png"
    ),
    dpi=500,
    bbox_inches="tight",
)

fig.savefig(
    os.path.join(
        OUTDIR,
        "Figure2_synthetic_main.pdf"
    ),
    bbox_inches="tight",
)

plt.close()