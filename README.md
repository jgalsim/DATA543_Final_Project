# North Carolina Neighborhood Flood Risk Project

This project builds a tract-level modeling dataset for North Carolina flood risk using a hazard, exposure, and vulnerability framework. It is designed to be robust in Google Colab, including cases where NLCD or 3DEP services are temporarily unavailable.

## What the project does

1. Downloads FEMA NFIP claims for North Carolina from OpenFEMA.
2. Downloads North Carolina census tract boundaries from TIGER/Line.
3. Downloads ACS tract-level demographic variables from the Census API.
4. Attempts to download NLCD imperviousness and 3DEP elevation and slope.
5. Aggregates claims to tracts.
6. Builds a tract-level modeling table.
7. Creates quick maps when the needed fields are available.
8. Runs baseline regression and classification models.
9. Runs a simple imperviousness reduction scenario when imperviousness is available.

## Folder structure

- `run_pipeline.py` runs the workflow end to end.
- `src/config.py` stores settings and paths.
- `src/acquire.py` downloads and saves raw data.
- `src/features.py` builds tract-level variables and maps.
- `src/model.py` runs the modeling workflow.
- `data/raw/` stores downloaded data.
- `data/processed/` stores the final model table.
- `outputs/` stores maps and model metrics.

## Colab setup

```python
from google.colab import files
uploaded = files.upload()
```

Upload `flood_risk_project_fixed.zip`, then run:

```python
!unzip -o flood_risk_project_fixed.zip
%cd /content/flood_risk_project_fixed
!pip install -r requirements.txt
!python run_pipeline.py
```

## Start analysis in Colab

```python
import pandas as pd
import geopandas as gpd

model_df = pd.read_csv("data/processed/nc_flood_model_table.csv")
model_df.head()
```

```python
model_df[["claim_count", "claim_rate_per_1000_units", "impervious_mean", "median_income"]].describe()
```

```python
model_df[["claim_rate_per_1000_units", "impervious_mean", "median_income", "pct_renter", "pct_elderly"]].corr(numeric_only=True)
```

## Notes

- FEMA claims are approximate and best treated as a tract-level proxy.
- If elevation or slope downloads fail because the external service is unavailable, the workflow still continues.
- You can later add floodplain overlays, CDC SVI, stream distance, parcels, roads, and NOAA rainfall.
