import os
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
INDIR = r"C:\Users\jenkints\Documents\GitHub\serval-fishsim-smk\output\paper_new"

OUTDIR = os.path.join(INDIR, "manuscript_summary_tables")
os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
reviewer_summary_df = pd.read_csv(
    os.path.join(
        INDIR,
        "reviewer_transcript_level_metrics",
        "reviewer_transcript_level_summary_by_replicate.csv",
    )
)

bulk_df = pd.read_csv(
    os.path.join(INDIR, "bulk_metrics.tsv.gz"),
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
# Compute max Pearson r per run x replicate x decoder
# --------------------------------------------------
idx = (
    bulk_df
    .dropna(subset=["r"])
    .groupby(["run", "replicate", "decoder"])["r"]
    .idxmax()
)

max_r_df = bulk_df.loc[
    idx,
    ["run", "replicate", "decoder", "r"]
].copy()

max_r_df = max_r_df.rename(columns={"r": "max_r"})

# --------------------------------------------------
# Merge transcript-level and abundance metrics
# --------------------------------------------------
df = reviewer_summary_df.merge(
    max_r_df,
    on=["run", "replicate", "decoder"],
    how="inner",
)

# --------------------------------------------------
# Optional: exclude Scenario 11 from primary benchmark
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
# Metrics to summarize
# --------------------------------------------------
metrics = [
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
        "simulation_primary_metrics_summary_numeric.csv"
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

# Optional ordering
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
        "simulation_primary_metrics_manuscript_table.csv"
    ),
    index=False,
)

print("Saved tables to:", OUTDIR)
print(paper_table)