"""Tests for the arrest demographics disparity analysis."""

import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Initialize Ray for Modin
os.environ["MODIN_ENGINE"] = "ray"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ray
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

import modin.pandas as pd


class TestRRRAnalysis:
    """Tests for RRR calculation functions."""
    
    def test_arrest_proportions_sum_to_one(self):
        """Test that arrest proportions sum to approximately 1 for each offense."""
        from rrr_analysis import calculate_arrest_proportions
        
        # Create test data
        test_df = pd.DataFrame({
            "offense": ["Test Crime"],
            "White": [600],
            "Black": [300],
            "American_Indian": [50],
            "Asian": [40],
            "Pacific_Islander": [10],
        })
        
        result = calculate_arrest_proportions(test_df)
        
        # Sum the proportions
        prop_cols = [c for c in result.columns if c.endswith("_prop")]
        total = sum(float(result[c].iloc[0]) for c in prop_cols)
        
        assert abs(total - 1.0) < 0.001, f"Proportions should sum to 1, got {total}"
    
    def test_rrr_parity(self):
        """Test that RRR = 1 when arrest proportion equals population proportion."""
        from rrr_analysis import compute_rrr
        
        # Arrest proportions
        arrest_df = pd.DataFrame({
            "offense": ["Test Crime"],
            "White_prop": [0.6],
            "Black_prop": [0.3],
            "total_arrests": [1000],
        })
        
        # Population proportions (same as arrest proportions)
        pop_df = pd.DataFrame({
            "race": ["White", "Black"],
            "population": [60000, 30000],
            "proportion": [0.6, 0.3],
        })
        
        result = compute_rrr(arrest_df, pop_df)
        
        # RRR should be 1.0 for both races
        assert abs(float(result["White_RRR"].iloc[0]) - 1.0) < 0.001
        assert abs(float(result["Black_RRR"].iloc[0]) - 1.0) < 0.001
    
    def test_rrr_over_representation(self):
        """Test that RRR > 1 indicates over-representation."""
        from rrr_analysis import compute_rrr
        
        # Group X is 50% of arrests but only 10% of population
        arrest_df = pd.DataFrame({
            "offense": ["Test Crime"],
            "White_prop": [0.5],
            "total_arrests": [1000],
        })
        
        pop_df = pd.DataFrame({
            "race": ["White"],
            "population": [10000],
            "proportion": [0.1],
        })
        
        result = compute_rrr(arrest_df, pop_df)
        
        # RRR should be 5.0 (0.5 / 0.1)
        assert abs(float(result["White_RRR"].iloc[0]) - 5.0) < 0.001


class TestClustering:
    """Tests for clustering functions."""
    
    def test_prepare_features_shape(self):
        """Test that feature matrix has correct shape."""
        from clustering import prepare_features
        
        rrr_df = pd.DataFrame({
            "offense": ["Crime1", "Crime2", "Crime3"],
            "White_RRR": [1.0, 2.0, 0.5],
            "Black_RRR": [2.0, 1.0, 1.5],
        })
        
        features, labels = prepare_features(rrr_df)
        
        assert features.shape == (3, 2), f"Expected (3, 2), got {features.shape}"
        assert len(labels) == 3
    
    def test_kmedoids_returns_correct_clusters(self):
        """Test K-Medoids returns correct number of clusters."""
        from clustering import cluster_kmedoids
        
        # Create well-separated clusters
        features = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0],
            [5, 5], [5.1, 5], [5, 5.1],
        ])
        
        result = cluster_kmedoids(features, n_clusters=2)
        
        assert result.n_clusters == 2
        assert len(set(result.labels)) == 2
    
    def test_gmm_probabilities_sum_to_one(self):
        """Test that GMM probabilities sum to 1 for each sample."""
        from clustering import cluster_gmm
        
        features = np.random.randn(10, 3)
        result = cluster_gmm(features, n_clusters=2)
        
        prob_sums = result.probabilities.sum(axis=1)
        
        for i, s in enumerate(prob_sums):
            assert abs(s - 1.0) < 0.001, f"Probabilities should sum to 1, got {s} for sample {i}"
    
    def test_hierarchical_linkage_matrix(self):
        """Test that hierarchical clustering produces valid linkage matrix."""
        from clustering import cluster_hierarchical
        
        features = np.random.randn(5, 2)
        result = cluster_hierarchical(features, n_clusters=2)
        
        # Linkage matrix should have n-1 rows and 4 columns
        assert result.linkage_matrix.shape == (4, 4)


class TestDataProcessing:
    """Tests for data processing functions."""
    
    def test_population_proportions_sum_to_less_than_one(self):
        """Test that race proportions sum to less than 1 (not all races included)."""
        from data_processing import parse_census_population, calculate_population_proportions
        
        census_data = {
            "Total": 100000,
            "White_alone_not_Hispanic": 60000,
            "Black_or_African_American_alone": 13000,
            "Asian_alone": 6000,
        }
        
        pop_df = parse_census_population(census_data)
        result = calculate_population_proportions(pop_df)
        
        total_prop = result["proportion"].sum()
        
        # Should be less than 1 since we don't include all races
        assert total_prop < 1.0
        assert total_prop > 0.5  # But should be a reasonable portion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
