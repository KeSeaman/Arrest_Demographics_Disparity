"""Clustering module for crime demographic profiles.

Implements three clustering methods:
1. K-Medoids (PAM) - Robust to outliers, uses actual crimes as centers
2. Gaussian Mixture Models (GMM) - Soft clustering with probability assignments
3. Hierarchical Clustering (Agglomerative) - Builds dendrogram taxonomy

Uses Modin Pandas for distributed DataFrame processing.
"""

from dataclasses import dataclass
from typing import Final
import os

os.environ["MODIN_ENGINE"] = "ray"

import numpy as np
import modin.pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DEFAULT_N_CLUSTERS: Final[int] = 4


@dataclass
class ClusteringResult:
    """Container for clustering results."""
    labels: np.ndarray
    n_clusters: int
    method: str
    centers: np.ndarray | None = None
    probabilities: np.ndarray | None = None  # For GMM
    linkage_matrix: np.ndarray | None = None  # For hierarchical
    medoid_indices: np.ndarray | None = None  # For K-Medoids


def prepare_features(rrr_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Prepare feature matrix from RRR DataFrame.
    
    Args:
        rrr_df: DataFrame with RRR values per crime.
        
    Returns:
        Tuple of (feature_matrix, crime_labels).
    """
    rrr_cols = [c for c in rrr_df.columns if c.endswith("_RRR")]
    
    # Extract features and handle nulls - convert to pandas for numpy ops
    features_df = rrr_df[rrr_cols].fillna(1.0)
    
    # Convert Modin to pandas then to numpy
    if hasattr(features_df, '_to_pandas'):
        features = features_df._to_pandas().values
    else:
        features = features_df.values
    
    crime_labels = rrr_df["offense"].tolist()
    
    # Log-transform to reduce skewness (RRR values can be very large)
    features = np.log1p(features)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, crime_labels


def _find_medoids(features: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Find actual data point closest to each cluster center (medoid).
    
    Args:
        features: Feature matrix.
        labels: Cluster assignments.
        n_clusters: Number of clusters.
        
    Returns:
        Array of medoid indices.
    """
    medoid_indices = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_points = features[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_points) > 0:
            # Find centroid
            centroid = cluster_points.mean(axis=0)
            # Find point closest to centroid
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            medoid_idx = cluster_indices[np.argmin(distances)]
            medoid_indices.append(medoid_idx)
    
    return np.array(medoid_indices)


def cluster_kmedoids(
    features: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 42,
) -> ClusteringResult:
    """Perform K-Medoids-style clustering using K-Means + medoid finding.
    
    K-Medoids is more robust to outliers than K-Means because it uses
    actual data points as cluster centers (medoids).
    
    Args:
        features: Feature matrix (n_samples, n_features).
        n_clusters: Number of clusters.
        random_state: Random seed for reproducibility.
        
    Returns:
        ClusteringResult with labels and medoid indices.
    """
    # Use KMeans for initial clustering
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(features)
    
    # Find medoids (actual data points closest to centroids)
    medoid_indices = _find_medoids(features, labels, n_clusters)
    
    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        method="K-Medoids (PAM-approximation)",
        centers=model.cluster_centers_,
        medoid_indices=medoid_indices,
    )


def cluster_gmm(
    features: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 42,
) -> ClusteringResult:
    """Perform Gaussian Mixture Model clustering.
    
    GMM provides soft clustering - each crime gets a probability of
    belonging to each cluster, not just a hard assignment.
    
    Args:
        features: Feature matrix (n_samples, n_features).
        n_clusters: Number of mixture components.
        random_state: Random seed for reproducibility.
        
    Returns:
        ClusteringResult with labels and probability matrix.
    """
    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_state,
        n_init=5,
    )
    
    labels = model.fit_predict(features)
    probabilities = model.predict_proba(features)
    
    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        method="Gaussian Mixture Model",
        centers=model.means_,
        probabilities=probabilities,
    )


def cluster_hierarchical(
    features: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    method: str = "ward",
) -> ClusteringResult:
    """Perform hierarchical (agglomerative) clustering.
    
    Builds a dendrogram tree showing the taxonomy of crimes based on
    their demographic profiles.
    
    Args:
        features: Feature matrix (n_samples, n_features).
        n_clusters: Number of clusters to cut the tree at.
        method: Linkage method ('ward', 'complete', 'average', 'single').
        
    Returns:
        ClusteringResult with labels and linkage matrix.
    """
    # Compute linkage matrix
    linkage_matrix = linkage(features, method=method)
    
    # Cut the tree to get cluster labels
    labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust") - 1  # 0-indexed
    
    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        method=f"Hierarchical ({method})",
        linkage_matrix=linkage_matrix,
    )


def get_cluster_profiles(
    rrr_df: pd.DataFrame,
    labels: np.ndarray,
) -> dict[int, pd.DataFrame]:
    """Get the RRR profiles for each cluster.
    
    Args:
        rrr_df: DataFrame with RRR values.
        labels: Cluster labels for each crime.
        
    Returns:
        Dictionary mapping cluster ID to DataFrame of crimes in that cluster.
    """
    rrr_with_labels = rrr_df.copy()
    rrr_with_labels["cluster"] = labels
    
    profiles = {}
    for cluster_id in range(max(labels) + 1):
        cluster_df = rrr_with_labels[rrr_with_labels["cluster"] == cluster_id]
        profiles[cluster_id] = cluster_df
    
    return profiles


def describe_clusters(
    rrr_df: pd.DataFrame,
    labels: np.ndarray,
    crime_labels: list[str],
) -> list[dict]:
    """Generate human-readable descriptions of each cluster.
    
    Args:
        rrr_df: DataFrame with RRR values.
        labels: Cluster labels.
        crime_labels: Names of crimes.
        
    Returns:
        List of cluster description dictionaries.
    """
    rrr_cols = [c for c in rrr_df.columns if c.endswith("_RRR")]
    rrr_with_labels = rrr_df.copy()
    rrr_with_labels["cluster"] = labels
    
    descriptions = []
    for cluster_id in range(max(labels) + 1):
        cluster_df = rrr_with_labels[rrr_with_labels["cluster"] == cluster_id]
        
        # Get mean RRR for each race in this cluster
        mean_rrrs = {}
        for col in rrr_cols:
            mean_val = cluster_df[col].mean()
            # Handle Modin series
            if hasattr(mean_val, 'item'):
                mean_val = mean_val.item() if not pd.isna(mean_val) else 0
            mean_rrrs[col.replace("_RRR", "")] = mean_val if mean_val else 0
        
        # Find crimes in this cluster
        crimes_in_cluster = [
            crime_labels[i] for i, label in enumerate(labels) if label == cluster_id
        ]
        
        # Determine dominant demographic
        dominant_race = max(mean_rrrs, key=lambda k: mean_rrrs[k])
        
        descriptions.append({
            "cluster_id": cluster_id,
            "n_crimes": len(crimes_in_cluster),
            "crimes": crimes_in_cluster,
            "mean_rrrs": mean_rrrs,
            "dominant_demographic": dominant_race,
            "dominant_rrr": mean_rrrs[dominant_race],
        })
    
    return descriptions


if __name__ == "__main__":
    from data_acquisition import fetch_all_data
    from data_processing import parse_fbi_excel, parse_census_population, calculate_population_proportions
    from rrr_analysis import calculate_arrest_proportions, compute_rrr
    
    # Test the functions
    fbi_path, census_data = fetch_all_data()
    fbi_df = parse_fbi_excel(fbi_path)
    pop_df = parse_census_population(census_data)
    pop_df = calculate_population_proportions(pop_df)
    
    arrest_props = calculate_arrest_proportions(fbi_df)
    rrr_df = compute_rrr(arrest_props, pop_df)
    
    # Prepare features
    features, crime_labels = prepare_features(rrr_df)
    print(f"Feature matrix shape: {features.shape}")
    
    # Test each clustering method
    print("\n=== K-Medoids Clustering ===")
    kmedoids_result = cluster_kmedoids(features)
    print(f"Labels: {kmedoids_result.labels}")
    
    print("\n=== GMM Clustering ===")
    gmm_result = cluster_gmm(features)
    print(f"Labels: {gmm_result.labels}")
    print(f"Sample probabilities:\n{gmm_result.probabilities[:3]}")
    
    print("\n=== Hierarchical Clustering ===")
    hier_result = cluster_hierarchical(features)
    print(f"Labels: {hier_result.labels}")
    
    # Describe clusters
    print("\n=== Cluster Descriptions ===")
    for desc in describe_clusters(rrr_df, kmedoids_result.labels, crime_labels):
        print(f"\nCluster {desc['cluster_id']} ({desc['n_crimes']} crimes):")
        print(f"  Dominant: {desc['dominant_demographic']} (RRR={desc['dominant_rrr']:.2f})")
        print(f"  Crimes: {desc['crimes'][:5]}...")
