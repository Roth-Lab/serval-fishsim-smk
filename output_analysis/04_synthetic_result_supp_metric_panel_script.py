import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Paths
# --------------------------------------------------
RESULTS_DIR = r"/projects/molonc/scratch/aroth/projects/serval/results/fishsim/paper_100"

ANALYSIS_DIR = "/projects/molonc/scratch/jtsui/serval-fishsim-smk/output/paper_100"

OUTDIR = os.path.join(
    ANALYSIS_DIR,
    "figure2_synthetic_supp_metrics"
)

os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
reviewer_summary_df = pd.read_csv(
    os.path.join(
        ANALYSIS_DIR,
        "reviewer_transcript_level_metrics",
        "reviewer_transcript_level_summary_by_replicate.csv",
    )
)

bulk_df = pd.read_csv(
    os.path.join(RESULTS_DIR, "bulk_metrics.tsv.gz"),
    sep="\t",
)

# --------------------------------------------------
# Remove old cosine decoder
# --------------------------------------------------
reviewer_summary_df = reviewer_summary_df[
    reviewer_summary_df["decoder"] != "cosine"
].copy()

bulk_df = bulk_df[
    bulk_df["decoder"] != "cosine"
].copy()

# --------------------------------------------------
# Compute max Spearman rho
# --------------------------------------------------
idx_rho = (
    bulk_df
    .dropna(subset=["rho"])
    .groupby(["run", "replicate", "decoder"])["rho"]
    .idxmax()
)

max_rho_df = bulk_df.loc[
    idx_rho,
    ["run", "replicate", "decoder", "rho", "threshold", "num_emitters"]
].copy()

max_rho_df = max_rho_df.rename(columns={
    "rho": "max_rho",
    "threshold": "threshold_at_max_rho",
    "num_emitters": "num_emitters_at_max_rho",
})

# --------------------------------------------------
# Merge transcript-level metrics with max Spearman rho
# --------------------------------------------------
fig_df = reviewer_summary_df.merge(
    max_rho_df,
    on=["run", "replicate", "decoder"],
    how="inner",
)

# --------------------------------------------------
# Clean decoder labels
# --------------------------------------------------
decoder_labels = {
    "cosine_np": "Cosine",
    "cosine-np": "Cosine",
    "scaled": "MERlin-scaled",
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

fig_df["decoder_label"] = (
    fig_df["decoder"]
    .astype(str)
    .map(decoder_labels)
    .fillna(fig_df["decoder"].astype(str))
)

# --------------------------------------------------
# Scenario order
# --------------------------------------------------
def scenario_sort_key(x):
    digits = "".join([c for c in str(x) if c.isdigit()])
    return int(digits) if digits else 999

run_order = sorted(fig_df["run"].unique(), key=scenario_sort_key)

# Exclude Scenario 11 from displayed benchmark
run_order = [r for r in run_order if scenario_sort_key(r) != 11]

fig_df = fig_df[fig_df["run"].isin(run_order)].copy()

# --------------------------------------------------
# Metrics for supplementary synthetic panel
# --------------------------------------------------
plot_metrics = [
    "max_rho",
    "exc_average_precision",
    "loc_max_f1",
    "loc_recall_at_max_f1",
]

metric_labels = {
    "max_rho": "Max Spearman rho",
    "exc_average_precision": "Exact average precision",
    "loc_max_f1": "Localization F1",
    "loc_recall_at_max_f1": "Localization recall",
}

panel_titles = {
    "max_rho": "A. Rank-based abundance recovery",
    "exc_average_precision": "B. Exact average precision",
    "loc_max_f1": "C. Localization F1",
    "loc_recall_at_max_f1": "D. Localization recall",
}

# --------------------------------------------------
# Check missing columns
# --------------------------------------------------
missing = [m for m in plot_metrics if m not in fig_df.columns]
if missing:
    raise ValueError(f"Missing metric columns: {missing}")

# --------------------------------------------------
# Make 2x2 supplementary figure
# --------------------------------------------------
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 9),
    sharex=False,
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

    if i == 0:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.tick_params(axis="x", rotation=45)

    ax.set_ylim(0, 1.05)

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
        "Figure2_synthetic_supp_metrics.png",
    ),
    dpi=500,
    bbox_inches="tight",
)

fig.savefig(
    os.path.join(
        OUTDIR,
        "Figure2_synthetic_supp_metrics.pdf",
    ),
    bbox_inches="tight",
)

plt.close()

print("Saved supplementary synthetic metric figure to:", OUTDIR)