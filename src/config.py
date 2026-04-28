from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

STATE_CODE = "NC"
STATE_FIPS = "37"
ACS_YEAR = 2023
NLCD_YEAR = 2021
NC_BBOX = (-84.32, 33.84, -75.46, 36.59)
TRACTS_URL = f"https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_{STATE_FIPS}_tract.zip"

ACS_VARIABLES = {
    "population": "B01003_001E",
    "housing_units": "B25001_001E",
    "median_income": "B19013_001E",
    "owner_occupied": "B25003_002E",
    "renter_occupied": "B25003_003E",
    "age_65_66": "B01001_020E",
    "age_67_69": "B01001_021E",
    "age_70_74": "B01001_022E",
    "age_75_79": "B01001_023E",
    "age_80_84": "B01001_024E",
    "age_85_plus_m": "B01001_025E",
    "age_65_66_f": "B01001_044E",
    "age_67_69_f": "B01001_045E",
    "age_70_74_f": "B01001_046E",
    "age_75_79_f": "B01001_047E",
    "age_80_84_f": "B01001_048E",
    "age_85_plus_f": "B01001_049E",
    "no_vehicle": "B08201_002E",
    "total_households_vehicle": "B08201_001E",
}

for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)
