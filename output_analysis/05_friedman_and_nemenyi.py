import os
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------
RESULTS_DIR = r"/projects/molonc/scratch/jtsui/serval-fishsim-smk-with-savannah/results/paper_100"

ANALYSIS_DIR = "/projects/molonc/scratch/jtsui/serval-fishsim-smk-with-savannah/output/paper_100"

OUTDIR = os.path.join(
    ANALYSIS_DIR,
    "manuscript_summary_tables"
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
# Merge metrics
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
# Exclude Scenario 11
# --------------------------------------------------
df = df[df["run"] != "scenario11"].copy()

# --------------------------------------------------
# Metrics to test: 6 out of 8
# --------------------------------------------------
metrics = [
    "exc_recall_at_max_f1",
    "exc_fdr_at_max_f1",
    "exc_max_f1",
    "max_r",
    "max_rho",
    "exc_average_precision",
]

metric_labels = {
    "exc_recall_at_max_f1": "Exact recall",
    "exc_fdr_at_max_f1": "Exact FDR",
    "exc_max_f1": "Exact F1",
    "max_r": "Max Pearson r",
    "max_rho": "Max Spearman rho",
    "exc_average_precision": "Exact average precision",
}

# --------------------------------------------------
# Check missing columns
# --------------------------------------------------
missing = [m for m in metrics if m not in df.columns]
if missing:
    raise ValueError(f"Missing metric columns: {missing}")

# --------------------------------------------------
# Multiple testing setup
# --------------------------------------------------
ALPHA = 0.01
N_TESTED_METRICS = len(metrics)

print(f"Family-wise alpha = {ALPHA}")
print(f"Number of tested metrics = {N_TESTED_METRICS}")

friedman_results = []
nemenyi_results = []

# Get the list of unique scenarios (S1 through S10)
scenarios = sorted(df["run"].unique())

# --------------------------------------------------
# Outer Loop: Run tests completely isolated by Scenario
# --------------------------------------------------
for scenario in scenarios:
    print("\n" + "#" * 80)
    print(f" PROCESSING SCENARIO: {scenario} ")
    print("#" * 80)
    
    scenario_df = df[df["run"] == scenario].copy()

    for metric in metrics:
        tmp = scenario_df[
            ["replicate", "decoder", metric]
        ].dropna().copy()

        # Legitimate Block = individual replicate within this specific scenario
        tmp["block"] = tmp["replicate"].astype(str)

        # Pivot to make rows = 100 replicates, columns = decoders
        wide = tmp.pivot(
            index="block",
            columns="decoder",
            values=metric,
        ).dropna()
        
        # Guard check to ensure we have data to test
        if wide.shape[0] < 3:
            print(f"Skipping {metric} for {scenario}: insufficient data count.")
            continue

        # --------------------------------------------------
        # Execute Friedman Test per Scenario-Metric combination
        # --------------------------------------------------
        stat, p_raw = friedmanchisquare(
            *[wide[col] for col in wide.columns]
        )

        # Bonferroni adjustment tracks metric families within this scenario
        p_bonf = min(p_raw * N_TESTED_METRICS, 1.0)
        significant = p_bonf < ALPHA

        friedman_results.append({
            "scenario": scenario,
            "metric": metric,
            "metric_label": metric_labels[metric],
            "n_blocks": wide.shape[0], # Will correctly be up to 100
            "n_decoders": wide.shape[1],
            "friedman_statistic": stat,
            "p_raw": p_raw,
            "p_bonf": p_bonf,
            "significant_0.01": significant,
        })

        print(f"\n{metric_labels[metric]} in {scenario}:")
        print(f"  Friedman statistic = {stat:.4f} (Blocks/Reps = {wide.shape[0]})")
        print(f"  Raw p = {p_raw:.3e} | Bonferroni p = {p_bonf:.3e}")

        # --------------------------------------------------
        # Post-hoc Nemenyi Test (Self-contained)
        # --------------------------------------------------
        if significant:
            nem = sp.posthoc_nemenyi_friedman(wide)
            decoders = list(nem.columns)
            n_pairs = len(decoders) * (len(decoders) - 1) / 2

            for i in range(len(decoders)):
                for j in range(i + 1, len(decoders)):
                    d1 = decoders[i]
                    d2 = decoders[j]

                    p_nem_raw = nem.loc[d1, d2]
                    p_nem_bonf = min(p_nem_raw * n_pairs, 1.0)

                    # Save result if it clears our adjusted alpha threshold
                    nemenyi_results.append({
                        "scenario": scenario,
                        "metric": metric,
                        "metric_label": metric_labels[metric],
                        "decoder1": d1,
                        "decoder2": d2,
                        "p_raw": p_nem_raw,
                        "p_bonf": p_nem_bonf,
                        "significant_0.01": p_nem_bonf < ALPHA,
                    })

# --------------------------------------------------
# Export Results
# --------------------------------------------------
friedman_df = pd.DataFrame(friedman_results)
nemenyi_df = pd.DataFrame(nemenyi_results)

friedman_df.to_csv(
    os.path.join(OUTDIR, "friedman_results_per_scenario.csv"),
    index=False,
)

nemenyi_df.to_csv(
    os.path.join(OUTDIR, "nemenyi_results_per_scenario.csv"),
    index=False,
)

print("\n" + "=" * 80)
print("COMPLETED: Results exported successfully to summary files.")
print("=" * 80)
