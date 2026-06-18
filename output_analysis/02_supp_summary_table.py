import os
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
RESULTS_DIR = r"/projects/molonc/scratch/aroth/projects/serval/results/fishsim/paper_100"

ANALYSIS_DIR = "/projects/molonc/scratch/jtsui/serval-fishsim-smk/output/paper_100"

OUTDIR = rf"{ANALYSIS_DIR}/manuscript_summary_tables"
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
].copy()

max_r_df = max_r_df.rename(columns={"r": "max_r"})

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
].copy()

max_rho_df = max_rho_df.rename(columns={"rho": "max_rho"})

# --------------------------------------------------
# Merge transcript-level and abundance metrics
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
# Exclude Scenario 11 from primary benchmark
# --------------------------------------------------
df_primary = df[df["run"] != "scenario11"].copy()

# --------------------------------------------------
# Clean decoder labels
# --------------------------------------------------
decoder_labels = {
    "cosine-np": "Cosine",
    "cosine_np": "Cosine",
    "scaled": "MERlin-scaled",
    "nn": "Nearest neighbor",
    "bardensr": "BARDENSR",
    "deepcell-spots": "DeepCell-Spots",
    "deepcell": "DeepCell-Spots",
}

df_primary["decoder_label"] = (
    df_primary["decoder"]
    .astype(str)
    .map(decoder_labels)
    .fillna(df_primary["decoder"].astype(str))
)

# --------------------------------------------------
# Metrics to summarize: 4 main + 4 supplementary
# --------------------------------------------------
metrics = [
    "exc_recall_at_max_f1",
    "exc_fdr_at_max_f1",
    "exc_max_f1",
    "max_r",
    "max_rho",
    "exc_average_precision",
    "loc_max_f1",
    "loc_recall_at_max_f1",
]

metric_labels = {
    "exc_recall_at_max_f1": "Exact recall",
    "exc_fdr_at_max_f1": "Exact FDR",
    "exc_max_f1": "Exact F1",
    "max_r": "Max Pearson r",
    "max_rho": "Max Spearman rho",
    "exc_average_precision": "Exact average precision",
    "loc_max_f1": "Localization F1",
    "loc_recall_at_max_f1": "Localization recall",
}

# --------------------------------------------------
# Check missing columns
# --------------------------------------------------
missing = [m for m in metrics if m not in df_primary.columns]
if missing:
    raise ValueError(f"Missing metric columns: {missing}")

# --------------------------------------------------
# Full numeric summary table
# --------------------------------------------------
summary_numeric = (
    df_primary
    .groupby("decoder_label")[metrics]
    .agg(["mean", "std", "sem", "median"])
    .round(4)
)

summary_numeric.to_csv(
    os.path.join(
        OUTDIR,
        "simulation_all_metrics_summary_numeric.csv",
    )
)

# --------------------------------------------------
# Manuscript-style table: mean ± SD
# --------------------------------------------------
rows = []

for decoder, g in df_primary.groupby("decoder_label"):

    row = {"Decoder": decoder}

    for metric in metrics:
        mean = g[metric].mean()
        sd = g[metric].std()
        row[metric_labels[metric]] = f"{mean:.3f} ± {sd:.3f}"

    rows.append(row)

paper_table = pd.DataFrame(rows)

decoder_order = [
    "Cosine",
    "MERlin-scaled",
    "BARDENSR",
    "Nearest neighbor",
    "DeepCell-Spots",
]

paper_table["Decoder"] = pd.Categorical(
    paper_table["Decoder"],
    categories=decoder_order,
    ordered=True,
)

paper_table = paper_table.sort_values("Decoder")

paper_table.to_csv(
    os.path.join(
        OUTDIR,
        "simulation_all_metrics_manuscript_table.csv",
    ),
    index=False,
)

print("Saved tables to:", OUTDIR)
print(paper_table)