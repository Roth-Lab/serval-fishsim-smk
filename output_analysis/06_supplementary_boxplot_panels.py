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
    "figure2to5_synthetic_sideway_boxplot_panels"
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
    "bardensr": "BarDensr",
    "deepcell": "DeepCell-Spots",
    "deepcell-spots": "DeepCell-Spots",
}

df["decoder_label"] = (
    df["decoder"]
    .astype(str)
    .map(decoder_labels)
    .fillna(df["decoder"].astype(str))
)

# Vertical (top-to-bottom) order of decoders within each boxplot cell
decoder_order = [
    "Cosine",
    "MERlin",
    "BarDensr",
    "Nearest neighbor",
    "DeepCell-Spots",
]

# --------------------------------------------------
# Color mapping  --  matched to the main line-plot (Figure 2)
# --------------------------------------------------
# The main line plot uses matplotlib's default tab10 palette assigned in
# ALPHABETICAL decoder order (BARDENSR, Cosine, DeepCell-Spots,
# Nearest neighbor, MERlin-scaled). The hex codes below reproduce that exact
# mapping so the boxplots are colour-consistent with the main figure.
# NOTE: if your main figure was generated with a custom palette rather than
# tab10, replace these hex codes with the ones used there.
# Updated color mapping to match your specified scheme
decoder_palette = {
    "BarDensr":         "#d62728",  # Red
    "Cosine":           "#2ca02c",  # Green
    "DeepCell-Spots":   "#9467bd",  # Purple
    "MERlin":           "#1f77b4",  # Blue
    "Nearest neighbor": "#ff7f0e",  # Orange
}

# --------------------------------------------------
# Scenario order / labels
# --------------------------------------------------
run_order = [f"scenario{i}" for i in range(1, 11)]
run_order = [r for r in run_order if r in df["run"].unique()]

scenario_labels = {
    f"scenario{i}": f"S{i}"
    for i in range(1, 11)
}

df = df[df["run"].isin(run_order)].copy()

# Split scenarios into two halves of (up to) five: row of S1-S5, row of S6-S10
scenario_halves = [run_order[:5], run_order[5:10]]

# --------------------------------------------------
# Metric display labels
# --------------------------------------------------
metric_labels = {
    "exc_recall_at_max_f1": "Exact recall",
    "exc_fdr_at_max_f1": "Exact FDR",
    "exc_max_f1": "Exact F1",
    "exc_average_precision": "Exact AP",
    "max_r": "Pearson $r$",
    "max_rho": "Spearman $\\rho$",
    "loc_recall_at_max_f1": "Localization recall",
    "loc_max_f1": "Localization F1",
}

# --------------------------------------------------
# Figure definitions
# Each figure = TWO metrics. metric_top occupies the top two rows
# (S1-S5, then S6-S10); metric_bottom occupies the bottom two rows.
#
# Detection (4 metrics) is split across Figures 2 and 3.
# Swap which two detection metrics land in each figure here if desired.
# --------------------------------------------------
figures = {
    "figure2_detection_recall_fdr": {
        "title": "Detection metrics: recall and false discovery rate",
        "metric_top": "exc_recall_at_max_f1",
        "metric_bottom": "exc_fdr_at_max_f1",
    },
    "figure3_detection_f1_ap": {
        "title": "Detection metrics: F1 and average precision",
        "metric_top": "exc_max_f1",
        "metric_bottom": "exc_average_precision",
    },
    "figure4_abundance": {
        "title": "Abundance recovery metrics",
        "metric_top": "max_r",
        "metric_bottom": "max_rho",
    },
    "figure5_localization": {
        "title": "Localization metrics",
        "metric_top": "loc_recall_at_max_f1",
        "metric_bottom": "loc_max_f1",
    },
}

# --------------------------------------------------
# Check missing columns
# --------------------------------------------------
all_metrics = []
for fig_info in figures.values():
    all_metrics.extend([fig_info["metric_top"], fig_info["metric_bottom"]])

missing = [m for m in set(all_metrics) if m not in df.columns]
if missing:
    raise ValueError(f"Missing metric columns: {missing}")

# --------------------------------------------------
# Plot helper
#   Layout per figure: 4 rows x 5 columns
#     row 0 : metric_top,    scenarios S1-S5
#     row 1 : metric_top,    scenarios S6-S10
#     row 2 : metric_bottom, scenarios S1-S5
#     row 3 : metric_bottom, scenarios S6-S10
#   Decoder names are written (in their decoder colour) to the right of the
#   last column in every row.
# --------------------------------------------------
def make_figure(fig_key, fig_info):

    metrics_blocks = [fig_info["metric_top"], fig_info["metric_bottom"]]

    n_cols = 5
    n_rows = 2 * len(metrics_blocks)  # 4

    fig_width = 3.6 * n_cols          # ~18
    fig_height = 2.8 * n_rows         # ~11.2

    sns.set_context("paper", font_scale=1.15)
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=True,
        gridspec_kw={"hspace": 0.45, "wspace": 0.12},
    )

    for block_idx, metric in enumerate(metrics_blocks):
        for half_idx, scenarios in enumerate(scenario_halves):

            row = block_idx * 2 + half_idx

            for col_idx in range(n_cols):

                ax = axes[row][col_idx]

                # Some halves may have fewer than 5 scenarios (if a run is absent)
                if col_idx >= len(scenarios):
                    ax.axis("off")
                    continue

                run = scenarios[col_idx]

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
                    palette=decoder_palette,
                    dodge=False,
                    showfliers=False,
                    width=0.78,
                    linewidth=1.45,
                    saturation=1.0,  # render palette at full strength (match main fig)
                    ax=ax,
                )

                # Remove per-panel legend
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

                ax.set_xlim(0, 1.08)

                # Scenario label on top of every cell
                ax.set_title(
                    scenario_labels.get(run, run),
                    fontsize=12,
                    fontweight="bold",
                )

                # Metric label on the left of the first column of each row
                if col_idx == 0:
                    ax.set_ylabel(
                        metric_labels[metric],
                        fontsize=12,
                        fontweight="bold",
                    )
                else:
                    ax.set_ylabel("")

                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

                # X axis only on the bottom row of each 2-row metric block
                if half_idx == len(scenario_halves) - 1:
                    ax.set_xlabel("Metric value", fontsize=10)
                    ax.tick_params(axis="x", labelsize=8)
                else:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])
                    ax.tick_params(axis="x", labelsize=8)

                # Decoder names (in their colour) to the right of the last column
                if col_idx == n_cols - 1:
                    for y_idx, decoder in enumerate(decoder_order):
                        ax.text(
                            1.12,
                            y_idx,
                            decoder,
                            va="center",
                            ha="left",
                            fontsize=9,
                            fontweight="bold",
                            color=decoder_palette.get(decoder, "black"),
                            transform=ax.get_yaxis_transform(),
                            clip_on=False,
                        )

    fig.suptitle(
        fig_info["title"],
        fontsize=17,
        fontweight="bold",
        y=1.005,
    )

    # Leave room on the right for the decoder labels; manual spacing is set via
    # gridspec_kw above, so avoid tight_layout (which conflicts with the
    # transform-based right-side text and emits a warning).
    fig.subplots_adjust(left=0.06, right=0.90, top=0.93, bottom=0.07)

    png_path = os.path.join(OUTDIR, f"synthetic_sideway_boxplot_{fig_key}.png")
    pdf_path = os.path.join(OUTDIR, f"synthetic_sideway_boxplot_{fig_key}.pdf")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")  # vector copy for submission

    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


# --------------------------------------------------
# Generate the four figures
# --------------------------------------------------
for fig_key, fig_info in figures.items():
    make_figure(fig_key, fig_info)

print("Done. Saved figures to:", OUTDIR)