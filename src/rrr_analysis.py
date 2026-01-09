"""Relative Risk Ratio (RRR) analysis module.

Calculates the Relative Risk Ratio for each crime type and demographic group.
RRR = (% of Arrestees who are Group X) / (% of Population who are Group X)

An RRR of 1.0 means perfect parity. RRR > 1.0 indicates over-representation.

Uses Modin Pandas for distributed DataFrame processing.
"""

import os
os.environ["MODIN_ENGINE"] = "ray"

import modin.pandas as pd
import numpy as np
from typing import Final

RACE_COLUMNS: Final[list[str]] = [
    "White",
    "Black",
    "American_Indian",
    "Asian",
    "Pacific_Islander",
]


def calculate_arrest_proportions(arrest_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the proportion of arrestees in each demographic group per crime.
    
    Args:
        arrest_df: FBI arrest data with race columns and offense column.
        
    Returns:
        DataFrame with offense and proportion columns for each race.
    """
    # Get available race columns
    available_races = [r for r in RACE_COLUMNS if r in arrest_df.columns]
    
    # Calculate row totals (sum of all race columns)
    result = arrest_df.copy()
    result["row_total"] = result[available_races].sum(axis=1)
    
    # Calculate proportions for each race
    for race in available_races:
        result[f"{race}_prop"] = result[race] / result["row_total"]
    
    # Select relevant columns
    prop_cols = [f"{race}_prop" for race in available_races]
    result = result[["offense"] + prop_cols + ["row_total"]].copy()
    result = result.rename(columns={"row_total": "total_arrests"})
    
    return result


def compute_rrr(
    arrest_proportions_df: pd.DataFrame,
    population_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Relative Risk Ratio for each crime-demographic combination.
    
    RRR = (% arrestees in group) / (% population in group)
    
    Args:
        arrest_proportions_df: DataFrame with arrest proportions per crime.
        population_df: DataFrame with population proportions per race.
        
    Returns:
        DataFrame with RRR values for each crime and race.
    """
    # Convert population proportions to a dictionary for easy lookup
    pop_dict = dict(zip(population_df["race"], population_df["proportion"]))
    
    result = arrest_proportions_df.copy()
    
    # Get available race columns (those ending in _prop)
    prop_cols = [c for c in result.columns if c.endswith("_prop")]
    
    # Calculate RRR for each race
    for prop_col in prop_cols:
        race = prop_col.replace("_prop", "")
        if race in pop_dict and pop_dict[race] > 0:
            result[f"{race}_RRR"] = result[prop_col] / pop_dict[race]
    
    # Select relevant columns
    rrr_cols = [c for c in result.columns if c.endswith("_RRR")]
    result = result[["offense", "total_arrests"] + rrr_cols]
    
    return result


def get_rrr_matrix(rrr_df: pd.DataFrame) -> tuple[list[str], list[str], list[list[float]]]:
    """Convert RRR DataFrame to matrix format for heatmap visualization.
    
    Args:
        rrr_df: DataFrame with RRR values.
        
    Returns:
        Tuple of (offense_labels, race_labels, rrr_values_matrix)
    """
    offense_labels = rrr_df["offense"].tolist()
    
    # Get RRR columns
    rrr_cols = [c for c in rrr_df.columns if c.endswith("_RRR")]
    race_labels = [c.replace("_RRR", "") for c in rrr_cols]
    
    # Build matrix
    rrr_matrix = rrr_df[rrr_cols].fillna(0).values.tolist()
    
    return offense_labels, race_labels, rrr_matrix


def identify_high_disparity_crimes(
    rrr_df: pd.DataFrame,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Identify crimes with high racial disparity (RRR above threshold).
    
    Args:
        rrr_df: DataFrame with RRR values.
        threshold: RRR threshold to consider as high disparity. Default 2.0.
        
    Returns:
        DataFrame with crimes having at least one RRR above threshold.
    """
    rrr_cols = [c for c in rrr_df.columns if c.endswith("_RRR")]
    
    # Filter to rows where any RRR exceeds threshold
    mask = (rrr_df[rrr_cols] > threshold).any(axis=1)
    
    return rrr_df[mask].sort_values("offense")


def rank_crimes_by_disparity(rrr_df: pd.DataFrame) -> pd.DataFrame:
    """Rank crimes by maximum RRR (highest disparity first).
    
    Args:
        rrr_df: DataFrame with RRR values.
        
    Returns:
        DataFrame sorted by max disparity with additional summary columns.
    """
    rrr_cols = [c for c in rrr_df.columns if c.endswith("_RRR")]
    
    result = rrr_df.copy()
    result["max_RRR"] = result[rrr_cols].max(axis=1)
    result = result.sort_values("max_RRR", ascending=False)
    
    return result


if __name__ == "__main__":
    from data_acquisition import fetch_all_data
    from data_processing import parse_fbi_excel, parse_census_population, calculate_population_proportions
    
    # Test the functions
    fbi_path, census_data = fetch_all_data()
    
    fbi_df = parse_fbi_excel(fbi_path)
    pop_df = parse_census_population(census_data)
    pop_df = calculate_population_proportions(pop_df)
    
    print("\nCalculating arrest proportions...")
    arrest_props = calculate_arrest_proportions(fbi_df)
    print(arrest_props.head(5))
    
    print("\nComputing RRR...")
    rrr_df = compute_rrr(arrest_props, pop_df)
    print(rrr_df.head(10))
    
    print("\nHigh disparity crimes (RRR > 2.0):")
    high_disparity = identify_high_disparity_crimes(rrr_df)
    print(high_disparity)
