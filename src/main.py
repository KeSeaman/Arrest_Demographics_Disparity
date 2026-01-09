#!/usr/bin/env python3
"""Main entry point for Arrest Demographics Disparity Analysis.

This script runs the complete analysis pipeline:
1. Fetches FBI Table 43A and Census ACS 2019 data
2. Calculates Relative Risk Ratios (RRR) for each crime-demographic combination
3. Performs clustering analysis using K-Medoids, GMM, and Hierarchical methods
4. Generates visualizations (heatmap, dendrogram, cluster profiles)
5. Outputs summary report

Uses Modin Pandas with Ray for distributed processing.
"""

import sys
import os
from pathlib import Path

# Initialize Ray for Modin
os.environ["MODIN_ENGINE"] = "ray"

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Initialize Ray before other imports
import ray
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

import modin.pandas as pd

from data_acquisition import fetch_all_data
from data_processing import parse_fbi_excel, parse_census_population, calculate_population_proportions
from rrr_analysis import (
    calculate_arrest_proportions,
    compute_rrr,
    get_rrr_matrix,
    rank_crimes_by_disparity,
)
from clustering import (
    prepare_features,
    cluster_kmedoids,
    cluster_gmm,
    cluster_hierarchical,
    describe_clusters,
)
from visualization import create_all_visualizations


def run_analysis() -> dict:
    """Run the complete disparity analysis pipeline.
    
    Returns:
        Dictionary containing all analysis results.
    """
    print("=" * 60)
    print("ARREST DEMOGRAPHICS DISPARITY ANALYSIS")
    print("FBI 2019 UCR Data + Census ACS 2019")
    print("Using Modin Pandas with Ray backend")
    print("=" * 60)
    
    # Step 1: Data Acquisition
    print("\n[1/5] Fetching data...")
    fbi_path, census_data = fetch_all_data()
    
    # Step 2: Data Processing
    print("\n[2/5] Processing data...")
    fbi_df = parse_fbi_excel(fbi_path)
    pop_df = parse_census_population(census_data)
    pop_df = calculate_population_proportions(pop_df)
    
    print(f"  Loaded {len(fbi_df)} offense categories from FBI data")
    print(f"  Census population breakdown:")
    for idx in range(len(pop_df)):
        row = pop_df.iloc[idx]
        print(f"    {row['race']}: {int(row['population']):,} ({row['proportion']*100:.1f}%)")
    
    # Step 3: RRR Analysis
    print("\n[3/5] Calculating Relative Risk Ratios...")
    arrest_props = calculate_arrest_proportions(fbi_df)
    rrr_df = compute_rrr(arrest_props, pop_df)
    
    # Rank by disparity
    ranked_df = rank_crimes_by_disparity(rrr_df)
    print("\n  Top 10 crimes by highest disparity (max RRR):")
    for i in range(min(10, len(ranked_df))):
        row = ranked_df.iloc[i]
        rrr_cols = [c for c in ranked_df.columns if c.endswith("_RRR")]
        max_rrr_col = max(rrr_cols, key=lambda c: row[c] if row[c] else 0)
        max_race = max_rrr_col.replace("_RRR", "")
        max_val = row[max_rrr_col]
        print(f"    {i+1}. {row['offense']}: {max_race} RRR={max_val:.2f}")
    
    # Step 4: Clustering
    print("\n[4/5] Clustering crime types by demographic profile...")
    features, crime_labels = prepare_features(rrr_df)
    
    n_clusters = 4
    kmedoids_result = cluster_kmedoids(features, n_clusters=n_clusters)
    gmm_result = cluster_gmm(features, n_clusters=n_clusters)
    hier_result = cluster_hierarchical(features, n_clusters=n_clusters)
    
    print(f"\n  {kmedoids_result.method} cluster descriptions:")
    for desc in describe_clusters(rrr_df, kmedoids_result.labels, crime_labels):
        print(f"    Cluster {desc['cluster_id']}: {desc['n_crimes']} crimes")
        print(f"      Dominant: {desc['dominant_demographic']} (RRR={desc['dominant_rrr']:.2f})")
        print(f"      Sample crimes: {', '.join(desc['crimes'][:3])}")
    
    # Step 5: Visualization
    print("\n[5/5] Generating visualizations...")
    figure_paths = create_all_visualizations(
        rrr_df, kmedoids_result, gmm_result, hier_result, crime_labels
    )
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nGenerated {len(figure_paths)} figures:")
    for path in figure_paths:
        print(f"  - {path}")
    
    return {
        "fbi_df": fbi_df,
        "population_df": pop_df,
        "rrr_df": rrr_df,
        "ranked_df": ranked_df,
        "features": features,
        "crime_labels": crime_labels,
        "kmedoids_result": kmedoids_result,
        "gmm_result": gmm_result,
        "hier_result": hier_result,
        "figure_paths": figure_paths,
    }


def generate_report(results: dict, output_path: Path | None = None) -> Path:
    """Generate a markdown report summarizing the analysis.
    
    Args:
        results: Dictionary from run_analysis().
        output_path: Path to save the report.
        
    Returns:
        Path to the generated report.
    """
    output_path = output_path or Path(__file__).parent.parent / "outputs" / "reports" / "analysis_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rrr_df = results["rrr_df"]
    ranked_df = results["ranked_df"]
    kmedoids_result = results["kmedoids_result"]
    crime_labels = results["crime_labels"]
    
    lines = [
        "# Arrest Demographics Disparity Analysis Report",
        "",
        "## Overview",
        "",
        "This analysis examines FBI 2019 arrest data (Table 43A) to identify which crimes",
        "show the highest racial disproportion relative to the US population (2019 ACS Census).",
        "",
        "## Methodology",
        "",
        "- **Relative Risk Ratio (RRR)**: RRR = (% of Arrestees in Group) / (% of Population in Group)",
        "- RRR = 1.0 indicates parity with population",
        "- RRR > 1.0 indicates over-representation in arrests",
        "- RRR < 1.0 indicates under-representation in arrests",
        "",
        "## Top 10 Crimes by Highest Disparity",
        "",
        "| Rank | Crime | Most Over-Represented Group | RRR |",
        "|------|-------|----------------------------|-----|",
    ]
    
    for i in range(min(10, len(ranked_df))):
        row = ranked_df.iloc[i]
        rrr_cols = [c for c in ranked_df.columns if c.endswith("_RRR")]
        max_col = max(rrr_cols, key=lambda c: row[c] if row[c] else 0)
        lines.append(f"| {i+1} | {row['offense']} | {max_col.replace('_RRR', '')} | {row[max_col]:.2f} |")
    
    lines.extend([
        "",
        "## Clustering Analysis",
        "",
        f"Crimes were clustered into {kmedoids_result.n_clusters} groups using {kmedoids_result.method}.",
        "",
    ])
    
    for desc in describe_clusters(rrr_df, kmedoids_result.labels, crime_labels):
        lines.extend([
            f"### Cluster {desc['cluster_id']}: {desc['dominant_demographic']}-heavy crimes",
            "",
            f"- **N Crimes**: {desc['n_crimes']}",
            f"- **Dominant RRR**: {desc['dominant_rrr']:.2f}",
            f"- **Crimes**: {', '.join(desc['crimes'])}",
            "",
        ])
    
    lines.extend([
        "## Visualizations",
        "",
        "See the `outputs/figures/` directory for:",
        "",
        "1. **rrr_disparity_heatmap.png**: Log-scale heatmap of RRR by crime × race",
        "2. **crime_dendrogram.png**: Hierarchical clustering taxonomy",
        "3. **cluster_profiles.png**: Mean RRR per cluster",
        "4. **gmm_probabilities.png**: Soft cluster membership probabilities",
    ])
    
    output_path.write_text("\n".join(lines))
    print(f"\nReport saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    results = run_analysis()
    generate_report(results)
