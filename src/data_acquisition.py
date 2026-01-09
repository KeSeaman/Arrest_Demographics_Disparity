"""Data acquisition module for FBI UCR and Census Bureau data.

This module handles downloading FBI arrest statistics (Table 43A)
and fetching US population demographics from the Census Bureau API.
"""

from pathlib import Path
from typing import Final
import requests

# Constants
FBI_TABLE_43A_URL: Final[str] = (
    "https://ucr.fbi.gov/crime-in-the-u.s/2019/crime-in-the-u.s.-2019/"
    "tables/table-43/table-43a.xls/output.xls"
)

CENSUS_API_BASE: Final[str] = "https://api.census.gov/data/2019/acs/acs1"

# Census B03002 table: Hispanic or Latino Origin by Race
# These variables give us the race breakdown for the total US population
CENSUS_RACE_VARIABLES: Final[dict[str, str]] = {
    "B03002_001E": "Total",
    "B03002_003E": "White_alone_not_Hispanic",
    "B03002_004E": "Black_or_African_American_alone",
    "B03002_005E": "American_Indian_and_Alaska_Native_alone",
    "B03002_006E": "Asian_alone",
    "B03002_007E": "Native_Hawaiian_and_Other_Pacific_Islander_alone",
    "B03002_012E": "Hispanic_or_Latino",
}

DEFAULT_DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data" / "raw"


def fetch_fbi_table43(
    output_dir: Path | None = None,
    timeout: int = 60,
) -> Path:
    """Download FBI Table 43A (Arrests by Race and Ethnicity) Excel file.
    
    Args:
        output_dir: Directory to save the file. Defaults to data/raw/.
        timeout: Request timeout in seconds.
        
    Returns:
        Path to the downloaded Excel file.
        
    Raises:
        requests.RequestException: If download fails.
    """
    output_dir = output_dir or DEFAULT_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "table_43a.xls"
    
    # Check if already downloaded
    if output_path.exists():
        print(f"FBI Table 43A already exists at {output_path}")
        return output_path
    
    print(f"Downloading FBI Table 43A from {FBI_TABLE_43A_URL}...")
    
    response = requests.get(
        FBI_TABLE_43A_URL,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (research project)"},
    )
    response.raise_for_status()
    
    output_path.write_bytes(response.content)
    print(f"Saved FBI Table 43A to {output_path}")
    
    return output_path


def fetch_census_population(
    timeout: int = 30,
) -> dict[str, int]:
    """Fetch 2019 ACS population by race/ethnicity from Census Bureau API.
    
    Uses the B03002 table (Hispanic or Latino Origin by Race) to get
    population counts that align with FBI racial categories.
    
    Args:
        timeout: Request timeout in seconds.
        
    Returns:
        Dictionary mapping race category names to population counts.
        
    Raises:
        requests.RequestException: If API call fails.
        ValueError: If response format is unexpected.
    """
    variables = ",".join(CENSUS_RACE_VARIABLES.keys())
    url = f"{CENSUS_API_BASE}?get=NAME,{variables}&for=us:*"
    
    print(f"Fetching Census population data...")
    
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    
    data = response.json()
    
    # Response format: [[headers], [values]]
    if len(data) < 2:
        raise ValueError(f"Unexpected Census API response format: {data}")
    
    headers = data[0]
    values = data[1]
    
    # Create mapping from variable code to value
    raw_data = dict(zip(headers, values))
    
    # Map to readable names
    population = {
        readable_name: int(raw_data[var_code])
        for var_code, readable_name in CENSUS_RACE_VARIABLES.items()
        if var_code in raw_data
    }
    
    print(f"Fetched population data for {len(population)} categories")
    return population


def fetch_all_data(output_dir: Path | None = None) -> tuple[Path, dict[str, int]]:
    """Fetch both FBI and Census data.
    
    Args:
        output_dir: Directory to save FBI data.
        
    Returns:
        Tuple of (FBI Excel file path, Census population dict).
    """
    fbi_path = fetch_fbi_table43(output_dir)
    census_data = fetch_census_population()
    return fbi_path, census_data


if __name__ == "__main__":
    # Test the functions
    fbi_path, census_pop = fetch_all_data()
    print(f"\nFBI data saved to: {fbi_path}")
    print(f"\nCensus population data:")
    for race, count in census_pop.items():
        print(f"  {race}: {count:,}")
