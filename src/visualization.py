"""Visualization module for arrest disparity analysis.

Creates publication-quality visualizations including:
- RRR Heatmap (log-scale) showing disparity across crimes and demographics
- Dendrogram showing hierarchical clustering of crime types
- Cluster profile charts
- GMM probability distributions

Uses Modin Pandas for distributed DataFrame processing.
"""

from pathlib import Path
from typing import Final
import os

os.environ["MODIN_ENGINE"] = "ray"

import numpy as np
import modin.pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

from clustering import ClusteringResult

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
FIGURE_DPI: Final[int] = 150
DEFAULT_OUTPUT_DIR: Final[Path] = Path(__file__).parent.parent / "outputs" / "figures"


def setup_output_dir(output_dir: Path | None = None) -> Path:
    """Ensure output directory exists."""
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_rrr_heatmap(
    offenses: list[str],
    races: list[str],
    rrr_matrix: list[list[float]],
    output_dir: Path | None = None,
    filename: str = "rrr_disparity_heatmap.png",
) -> Path:
    """Create a log-scale heatmap of RRR values.
    
    Args:
        offenses: List of offense types (Y-axis labels).
        races: List of race categories (X-axis labels).
        rrr_matrix: 2D matrix of RRR values [n_offenses x n_races].
        output_dir: Directory to save the figure.
        filename: Output filename.
        
    Returns:
        Path to the saved figure.
    """
    output_dir = setup_output_dir(output_dir)
    
    # Convert to numpy array
    data = np.array(rrr_matrix)
    
    # Replace zeros/nulls with small value for log scale
    data = np.where(data <= 0, 0.01, data)
    data = np.where(np.isnan(data), 0.01, data)
    
    # Create figure with appropriate size
    n_offenses = len(offenses)
    fig_height = max(8, n_offenses * 0.4)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Use log scale for color normalization
    # RRR = 1 is parity, so center the colormap there
    norm = mcolors.LogNorm(vmin=0.1, vmax=10)
    
    # Create diverging colormap centered at 1.0 (parity)
    # Blue = under-representation, Red = over-representation
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Create heatmap
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    
    # Set tick labels
    ax.set_xticks(range(len(races)))
    ax.set_xticklabels(races, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(offenses)))
    ax.set_yticklabels(offenses, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Relative Risk Ratio (log scale)")
    cbar.ax.axhline(y=1.0, color="white", linewidth=2, linestyle="--")
    
    # Add value annotations for extreme values
    for i in range(len(offenses)):
        for j in range(len(races)):
            val = data[i, j]
            if val > 3.0 or val < 0.3:
                text_color = "white" if val > 3.0 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", 
                       color=text_color, fontsize=7, fontweight="bold")
    
    ax.set_title("Arrest Disparity by Crime Type and Race\n(RRR: 1.0 = Parity, >1 = Over-represented)", 
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Race/Ethnicity", fontsize=12)
    ax.set_ylabel("Offense Type", fontsize=12)
    
    plt.tight_layout()
    
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    
    print(f"Saved heatmap to {output_path}")
    return output_path


def plot_dendrogram(
    clustering_result: ClusteringResult,
    crime_labels: list[str],
    output_dir: Path | None = None,
    filename: str = "crime_dendrogram.png",
) -> Path:
    """Create a dendrogram visualization from hierarchical clustering.
    
    Args:
        clustering_result: Result from cluster_hierarchical().
        crime_labels: Labels for each crime type.
        output_dir: Directory to save the figure.
        filename: Output filename.
        
    Returns:
        Path to the saved figure.
    """
    output_dir = setup_output_dir(output_dir)
    
    if clustering_result.linkage_matrix is None:
        raise ValueError("ClusteringResult must have a linkage_matrix (from hierarchical clustering)")
    
    # Create figure
    n_labels = len(crime_labels)
    fig_height = max(10, n_labels * 0.3)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    # Plot dendrogram
    scipy_dendrogram(
        clustering_result.linkage_matrix,
        labels=crime_labels,
        orientation="right",
        leaf_font_size=9,
        ax=ax,
        color_threshold=0.7 * max(clustering_result.linkage_matrix[:, 2]),
    )
    
    ax.set_title("Crime Type Taxonomy Based on Demographic Profiles\n(Hierarchical Clustering)", 
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Distance (Ward linkage)", fontsize=12)
    
    plt.tight_layout()
    
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    
    print(f"Saved dendrogram to {output_path}")
    return output_path


def plot_cluster_profiles(
    rrr_df: pd.DataFrame,
    labels: np.ndarray,
    output_dir: Path | None = None,
    filename: str = "cluster_profiles.png",
) -> Path:
    """Create bar charts showing the mean RRR profile for each cluster.
    
    Args:
        rrr_df: DataFrame with RRR values.
        labels: Cluster labels for each crime.
        output_dir: Directory to save the figure.
        filename: Output filename.
        
    Returns:
        Path to the saved figure.
    """
    output_dir = setup_output_dir(output_dir)
    
    rrr_cols = [c for c in rrr_df.columns if c.endswith("_RRR")]
    race_names = [c.replace("_RRR", "") for c in rrr_cols]
    n_clusters = len(set(labels))
    
    # Calculate mean RRR for each cluster
    rrr_with_labels = rrr_df.copy()
    rrr_with_labels["cluster"] = labels
    
    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 5), sharey=True)
    if n_clusters == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(race_names)))
    
    for cluster_id, ax in enumerate(axes):
        cluster_data = rrr_with_labels[rrr_with_labels["cluster"] == cluster_id]
        mean_rrrs = []
        for col in rrr_cols:
            mean_val = cluster_data[col].mean()
            # Handle potential Modin/pandas differences
            if hasattr(mean_val, 'item'):
                mean_val = mean_val.item() if not np.isnan(mean_val) else 0
            mean_rrrs.append(mean_val if mean_val and not np.isnan(mean_val) else 0)
        
        bars = ax.bar(race_names, mean_rrrs, color=colors)
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="Parity (RRR=1)")
        
        ax.set_title(f"Cluster {cluster_id}\n({len(cluster_data)} crimes)", fontsize=11)
        ax.set_xlabel("Race/Ethnicity")
        if cluster_id == 0:
            ax.set_ylabel("Mean RRR")
        ax.tick_params(axis="x", rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, mean_rrrs):
            if val and val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    
    fig.suptitle("Demographic Profiles by Crime Cluster", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    
    print(f"Saved cluster profiles to {output_path}")
    return output_path


def plot_gmm_probabilities(
    clustering_result: ClusteringResult,
    crime_labels: list[str],
    top_n: int = 15,
    output_dir: Path | None = None,
    filename: str = "gmm_probabilities.png",
) -> Path:
    """Create stacked bar chart showing GMM soft cluster assignments.
    
    Args:
        clustering_result: Result from cluster_gmm() with probabilities.
        crime_labels: Labels for each crime type.
        top_n: Number of crimes to show (those with most ambiguous assignments).
        output_dir: Directory to save the figure.
        filename: Output filename.
        
    Returns:
        Path to the saved figure.
    """
    output_dir = setup_output_dir(output_dir)
    
    if clustering_result.probabilities is None:
        raise ValueError("ClusteringResult must have probabilities (from GMM clustering)")
    
    probs = clustering_result.probabilities
    n_clusters = probs.shape[1]
    
    # Find crimes with most "mixed" cluster membership
    # Entropy is highest when probabilities are evenly distributed
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    top_n = min(top_n, len(crime_labels))
    top_indices = np.argsort(entropy)[-top_n:][::-1]
    
    selected_crimes = [crime_labels[i] for i in top_indices]
    selected_probs = probs[top_indices]
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    bottom = np.zeros(len(selected_crimes))
    
    for cluster_id in range(n_clusters):
        heights = selected_probs[:, cluster_id]
        ax.barh(selected_crimes, heights, left=bottom, label=f"Cluster {cluster_id}",
                color=colors[cluster_id])
        bottom += heights
    
    ax.set_xlabel("Probability", fontsize=12)
    ax.set_title("GMM Soft Cluster Assignments\n(Crimes with most mixed profiles)", 
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    
    print(f"Saved GMM probabilities to {output_path}")
    return output_path


def create_all_visualizations(
    rrr_df: pd.DataFrame,
    kmedoids_result: ClusteringResult,
    gmm_result: ClusteringResult,
    hier_result: ClusteringResult,
    crime_labels: list[str],
    output_dir: Path | None = None,
) -> list[Path]:
    """Create all visualizations for the analysis.
    
    Args:
        rrr_df: DataFrame with RRR values.
        kmedoids_result: Result from K-Medoids clustering.
        gmm_result: Result from GMM clustering.
        hier_result: Result from hierarchical clustering.
        crime_labels: Labels for each crime.
        output_dir: Directory to save figures.
        
    Returns:
        List of paths to created figures.
    """
    from rrr_analysis import get_rrr_matrix
    
    output_dir = setup_output_dir(output_dir)
    paths = []
    
    # 1. RRR Heatmap
    offenses, races, matrix = get_rrr_matrix(rrr_df)
    paths.append(plot_rrr_heatmap(offenses, races, matrix, output_dir))
    
    # 2. Dendrogram
    paths.append(plot_dendrogram(hier_result, crime_labels, output_dir))
    
    # 3. Cluster profiles (using K-Medoids labels)
    paths.append(plot_cluster_profiles(rrr_df, kmedoids_result.labels, output_dir))
    
    # 4. GMM probabilities
    paths.append(plot_gmm_probabilities(gmm_result, crime_labels, output_dir=output_dir))
    
    return paths


if __name__ == "__main__":
    from data_acquisition import fetch_all_data
    from data_processing import parse_fbi_excel, parse_census_population, calculate_population_proportions
    from rrr_analysis import calculate_arrest_proportions, compute_rrr, get_rrr_matrix
    from clustering import prepare_features, cluster_kmedoids, cluster_gmm, cluster_hierarchical
    
    # Run full pipeline
    fbi_path, census_data = fetch_all_data()
    fbi_df = parse_fbi_excel(fbi_path)
    pop_df = parse_census_population(census_data)
    pop_df = calculate_population_proportions(pop_df)
    
    arrest_props = calculate_arrest_proportions(fbi_df)
    rrr_df = compute_rrr(arrest_props, pop_df)
    
    features, crime_labels = prepare_features(rrr_df)
    
    # Cluster
    kmedoids_result = cluster_kmedoids(features)
    gmm_result = cluster_gmm(features)
    hier_result = cluster_hierarchical(features)
    
    # Visualize
    paths = create_all_visualizations(
        rrr_df, kmedoids_result, gmm_result, hier_result, crime_labels
    )
    print(f"\nCreated {len(paths)} visualizations")
