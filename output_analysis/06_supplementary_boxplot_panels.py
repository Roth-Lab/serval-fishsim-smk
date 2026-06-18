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
    "figure2_synthetic_sideway_boxplot_panels"
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
# Remove old cosine and Scenario 11
# --------------------------------------------------
reviewer_summary_df = reviewer_summary_df[
    (reviewer_summary_df["decoder"] != "cosine")
    & (reviewer_summary_df["run"] != "scenario11")
].copy()

bulk_df = bulk_df[
    (bulk_df["decoder"] != "cosine")
    & (bulk_df["run"] != "scenario11")
].copy()

# --------------------------------------------------
# Compute max Pearson r
# --------------------------------------------------
idx_r = (
    bulk_df
    .dropna(subset=["r"])
    .groupby(["run", "replicate", "decoder"])["r"]
    .idxmax()
)

max_r_df = bulk_df.loc[
    idx_r,
    ["run", "replicate", "decoder", "r"]
].rename(columns={"r": "max_r"})

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
    ["run", "replicate", "decoder", "rho"]
].rename(columns={"rho": "max_rho"})

# --------------------------------------------------
# Merge all metrics
# --------------------------------------------------
df = (
    reviewer_summary_df
    .merge(
        max_r_df,
        on=["run", "replicate", "decoder"],
        how="inner",
    )
    .merge(
        max_rho_df,
        on=["run", "replicate", "decoder"],
        how="inner",
    )
)

# --------------------------------------------------
# Decoder labels
# --------------------------------------------------
decoder_labels = {
    "cosine_np": "Cosine",
    "cosine-np": "Cosine",
    "scaled": "MERlin",
    "nn": "Nearest neighbor",
    "bardensr": "BARDENSR",
    "deepcell": "DeepCell-Spots",
    "deepcell-spots": "DeepCell-Spots",
}

df["decoder_label"] = (
    df["decoder"]
    .astype(str)
    .map(decoder_labels)
    .fillna(df["decoder"].astype(str))
)

decoder_order = [
    "Cosine",
    "MERlin",
    "BARDENSR",
    "Nearest neighbor",
    "DeepCell-Spots",
]

# --------------------------------------------------
# Scenario order
# --------------------------------------------------
run_order = [f"scenario{i}" for i in range(1, 11)]
run_order = [r for r in run_order if r in df["run"].unique()]

scenario_labels = {
    f"scenario{i}": f"S{i}"
    for i in range(1, 11)
}

df = df[df["run"].isin(run_order)].copy()

# --------------------------------------------------
# Metric panels
# --------------------------------------------------
panels = {
    "detection": {
        "title": "Detection metrics",
        "metrics": [
            "exc_recall_at_max_f1",
            "exc_fdr_at_max_f1",
            "exc_max_f1",
            "exc_average_precision",
        ],
        "labels": {
            "exc_recall_at_max_f1": "Exact recall",
            "exc_fdr_at_max_f1": "Exact FDR",
            "exc_max_f1": "Exact F1",
            "exc_average_precision": "Exact AP",
        },
    },
    "abundance": {
        "title": "Abundance recovery metrics",
        "metrics": [
            "max_r",
            "max_rho",
        ],
        "labels": {
            "max_r": "Pearson $r$",
            "max_rho": "Spearman $\\rho$",
        },
    },
    "localization": {
        "title": "Localization metrics",
        "metrics": [
            "loc_recall_at_max_f1",
            "loc_max_f1",
        ],
        "labels": {
            "loc_recall_at_max_f1": "Localization recall",
            "loc_max_f1": "Localization F1",
        },
    },
}

# --------------------------------------------------
# Check missing columns
# --------------------------------------------------
all_metrics = []
for panel in panels.values():
    all_metrics.extend(panel["metrics"])

missing = [m for m in all_metrics if m not in df.columns]
if missing:
    raise ValueError(f"Missing metric columns: {missing}")

# --------------------------------------------------
# Plot helper
# --------------------------------------------------
def make_sideway_panel(panel_name, panel_info):

    metrics = panel_info["metrics"]
    metric_labels = panel_info["labels"]

    n_rows = len(metrics)
    n_cols = len(run_order)

    fig_width = 34
    fig_height = max(6, 2.9 * n_rows)

    sns.set_context("paper", font_scale=1.15)
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=True,
    )

    if n_rows == 1:
        axes = [axes]

    for row_idx, metric in enumerate(metrics):

        for col_idx, run in enumerate(run_order):

            ax = axes[row_idx][col_idx]

            sub = df[
                (df["run"] == run)
                & df[metric].notna()
            ].copy()

            sns.boxplot(
                data=sub,
                x=metric,
                y="decoder_label",
                hue="decoder_label",
                order=decoder_order,
                hue_order=decoder_order,
                dodge=False,
                showfliers=False,
                width=0.78,
                linewidth=1.45,
                ax=ax,
            )

            # Remove per-panel legend
            if ax.get_legend() is not None:
                ax.get_legend().remove()

            ax.set_xlim(0, 1.08)

            if row_idx == 0:
                ax.set_title(
                    scenario_labels.get(run, run),
                    fontsize=12,
                    fontweight="bold",
                )

            if col_idx == 0:
                ax.set_ylabel(
                    metric_labels[metric],
                    fontsize=11,
                    fontweight="bold",
                )
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            if row_idx == n_rows - 1:
                ax.set_xlabel("Metric value", fontsize=10)
            else:
                ax.set_xlabel("")

            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", length=0)

            # Label decoder names only on the right side of the last scenario column
            if col_idx == n_cols - 1:
                for y_idx, decoder in enumerate(decoder_order):
                    ax.text(
                        1.105,
                        y_idx,
                        decoder,
                        va="center",
                        ha="left",
                        fontsize=9,
                        transform=ax.get_yaxis_transform(),
                        clip_on=False,
                    )

    fig.suptitle(
        panel_info["title"],
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout(rect=[0, 0, 0.965, 0.98])

    png_path = os.path.join(
        OUTDIR,
        f"synthetic_sideway_boxplot_{panel_name}.png",
    )

    fig.savefig(
        png_path,
        dpi=700,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", png_path)


# --------------------------------------------------
# Generate individual PNG panels
# --------------------------------------------------
for panel_name, panel_info in panels.items():
    make_sideway_panel(panel_name, panel_info)

print("Done. Saved PNGs to:", OUTDIR)