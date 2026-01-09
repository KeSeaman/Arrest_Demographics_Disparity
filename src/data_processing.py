"""Data processing module for FBI and Census data.

Parses raw data files and transforms them into analysis-ready DataFrames.
Uses Modin Pandas for distributed DataFrame processing.
"""

from pathlib import Path
from typing import Final
import os

# Initialize Ray for Modin before importing
os.environ["MODIN_ENGINE"] = "ray"
import ray
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

import pandas as _pd  # Use standard pandas for Excel parsing
import modin.pandas as pd  # Use Modin for distributed processing

# Standardized race categories for analysis
RACE_CATEGORIES: Final[list[str]] = [
    "White",
    "Black",
    "American_Indian",
    "Asian",
    "Pacific_Islander",
]


def parse_fbi_excel(file_path: Path) -> pd.DataFrame:
    """Parse FBI Table 43A Excel file into a Modin DataFrame.
    
    Uses standard pandas for Excel parsing (more reliable), then converts to Modin.
    FBI Table 43A structure (after skiprows=5):
    - Row 0: Column headers (Offense charged, Total, White, Black..., Total, White... for percentages)
    - Row 1+: Data rows
    
    Columns 0-6 are absolute numbers, columns 7-12 are percentages. We want absolute numbers.
    
    Args:
        file_path: Path to the downloaded Table 43A Excel file.
        
    Returns:
        DataFrame with columns: offense, White, Black, American_Indian, Asian, Pacific_Islander, Total
    """
    # Read the Excel file
    df_raw = _pd.read_excel(
        file_path,
        sheet_name=0,
        skiprows=5,
        header=0,
    )
    
    # Row 0 of df_raw contains actual column names, data starts from row 1
    # Extract column names from row 0
    header_row = df_raw.iloc[0]
    df = df_raw.iloc[1:].copy()
    
    # Select columns by position (0=Offense, 1=Total, 2=White, 3=Black, 4=American Indian, 5=Asian, 6=Pacific Islander)
    # These are the absolute number columns, not percentages
    selected_cols = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]].copy()
    
    # Assign clean column names
    selected_cols.columns = ["offense", "Total", "White", "Black", "American_Indian", "Asian", "Pacific_Islander"]
    
    # Filter to only include rows with valid offense names
    selected_cols = selected_cols[selected_cols["offense"].notna()]
    selected_cols = selected_cols[selected_cols["offense"].astype(str).str.len() > 0]
    selected_cols = selected_cols[~selected_cols["offense"].astype(str).str.match(r"^[0-9]")]  # Exclude footnotes
    selected_cols = selected_cols[~selected_cols["offense"].astype(str).str.lower().str.startswith("total")]  # Exclude total rows
    
    # Clean offense names - remove trailing numbers (footnote references)
    selected_cols["offense"] = (
        selected_cols["offense"].astype(str)
        .str.strip()
        .str.replace(r"\d+$", "", regex=True)
        .str.strip()
    )
    
    # Convert numeric columns
    numeric_cols = ["White", "Black", "American_Indian", "Asian", "Pacific_Islander", "Total"]
    for col in numeric_cols:
        selected_cols[col] = _pd.to_numeric(selected_cols[col], errors="coerce")
    
    # Remove rows with all null values in numeric columns
    selected_cols = selected_cols.dropna(subset=numeric_cols, how="all")
    
    # Reset index
    selected_cols = selected_cols.reset_index(drop=True)
    
    # Convert to Modin DataFrame for distributed processing
    return pd.DataFrame(selected_cols)


def parse_census_population(census_data: dict[str, int]) -> pd.DataFrame:
    """Convert Census population dictionary to a DataFrame."""
    census_to_standard = {
        "White_alone_not_Hispanic": "White",
        "Black_or_African_American_alone": "Black",
        "American_Indian_and_Alaska_Native_alone": "American_Indian",
        "Asian_alone": "Asian",
        "Native_Hawaiian_and_Other_Pacific_Islander_alone": "Pacific_Islander",
    }
    
    records = []
    for census_name, standard_name in census_to_standard.items():
        if census_name in census_data:
            records.append({
                "race": standard_name,
                "population": census_data[census_name],
            })
    
    if "Total" in census_data:
        records.append({
            "race": "Total",
            "population": census_data["Total"],
        })
    
    return pd.DataFrame(records)


def calculate_population_proportions(population_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the proportion of total population for each race."""
    total_pop = population_df[population_df["race"] == "Total"]["population"].iloc[0]
    
    result = population_df[population_df["race"] != "Total"].copy()
    result["proportion"] = result["population"] / total_pop
    
    return result


def prepare_analysis_data(
    fbi_df: pd.DataFrame,
    census_data: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare FBI and Census data for RRR analysis."""
    pop_df = parse_census_population(census_data)
    pop_df = calculate_population_proportions(pop_df)
    return fbi_df, pop_df


if __name__ == "__main__":
    from data_acquisition import fetch_all_data
    
    fbi_path, census_data = fetch_all_data()
    
    print("\nParsing FBI data...")
    fbi_df = parse_fbi_excel(fbi_path)
    print(f"FBI data shape: {fbi_df.shape}")
    print(fbi_df.head(10))
    
    print("\nProcessing Census data...")
    pop_df = parse_census_population(census_data)
    pop_df = calculate_population_proportions(pop_df)
    print(pop_df)
