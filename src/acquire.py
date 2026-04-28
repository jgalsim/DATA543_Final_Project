from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

from src.config import ACS_VARIABLES, ACS_YEAR, NC_BBOX, NLCD_YEAR, RAW_DIR, STATE_CODE, TRACTS_URL

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "unc-flood-risk-project/1.0"})
TIMEOUT = 90


def _get_json(url: str, params: dict[str, Any] | None = None) -> dict | list:
    response = SESSION.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_fema_claims(state_code: str = STATE_CODE, page_size: int = 5000, max_pages: int = 100) -> gpd.GeoDataFrame:
    base_url = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
    all_rows: list[dict[str, Any]] = []
    skip = 0

    for _ in range(max_pages):
        payload = _get_json(
            base_url,
            params={
                "$filter": f"state eq '{state_code}'",
                "$top": page_size,
                "$skip": skip,
            },
        )
        rows = payload.get("FimaNfipClaims", [])
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        skip += page_size

    if not all_rows:
        raise RuntimeError("No FEMA claims were returned from the API.")

    df = pd.DataFrame(all_rows)
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"]).copy()

    possible_numeric = [
        "amountPaidOnBuildingClaim",
        "amountPaidOnContentsClaim",
        "amountPaidOnIncreasedCostOfComplianceClaim",
        "netBuildingPaymentAmount",
        "netContentsPaymentAmount",
    ]
    for col in possible_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )


def save_fema_claims(gdf: gpd.GeoDataFrame, path: Path | None = None) -> Path:
    path = path or (RAW_DIR / "nc_fema_claims.parquet")
    gdf.to_parquet(path, index=False)
    return path


def load_nc_tracts() -> gpd.GeoDataFrame:
    tracts = gpd.read_file(TRACTS_URL).to_crs("EPSG:4326")
    tracts["GEOID"] = tracts["GEOID"].astype(str)
    return tracts[["GEOID", "NAME", "COUNTYFP", "geometry"]]


def save_nc_tracts(gdf: gpd.GeoDataFrame, path: Path | None = None) -> Path:
    path = path or (RAW_DIR / "nc_tracts.parquet")
    gdf.to_parquet(path, index=False)
    return path


def fetch_acs_tract_data(year: int = ACS_YEAR) -> pd.DataFrame:
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {"get": ",".join(ACS_VARIABLES.values()), "for": "tract:*", "in": "state:37"}
    rows = _get_json(url, params=params)
    if not rows or len(rows) < 2:
        raise RuntimeError("No ACS data returned from Census API.")

    header = rows[0]
    data = pd.DataFrame(rows[1:], columns=header)
    inverse_map = {v: k for k, v in ACS_VARIABLES.items()}
    data = data.rename(columns=inverse_map)
    data["GEOID"] = data["state"] + data["county"] + data["tract"]

    numeric_cols = [c for c in data.columns if c not in {"state", "county", "tract", "GEOID"}]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def save_acs(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or (RAW_DIR / "nc_acs_tracts.parquet")
    df.to_parquet(path, index=False)
    return path


def _coerce_raster_like(obj: Any):
    if obj is None:
        return None
    if isinstance(obj, dict):
        if not obj:
            return None
        obj = next(iter(obj.values()))
    if hasattr(obj, "data_vars"):
        data_vars = list(obj.data_vars)
        if not data_vars:
            return None
        obj = obj[data_vars[0]]
    return obj


def fetch_nlcd_impervious():
    try:
        import pygeohydro as gh
    except Exception as e:
        print(f"Warning: pygeohydro import failed: {e}")
        return None

    region = gpd.GeoDataFrame(index=[0], geometry=[box(*NC_BBOX)], crs="EPSG:4326")
    try:
        nlcd = gh.nlcd_bygeom(region, resolution=100, years={"impervious": [NLCD_YEAR]})
        return _coerce_raster_like(nlcd)
    except Exception as e:
        print(f"Warning: impervious download failed: {e}")
        return None


def fetch_dem_and_slope() -> tuple[Any, Any]:
    try:
        import py3dep
    except Exception as e:
        print(f"Warning: py3dep import failed: {e}")
        return None, None

    xmin, ymin, xmax, ymax = NC_BBOX
    dem = None
    slope = None
    try:
        dem = py3dep.get_map("DEM", (xmin, ymin, xmax, ymax), resolution=300)
        dem = _coerce_raster_like(dem)
    except Exception as e:
        print(f"Warning: DEM download failed: {e}")

    try:
        slope = py3dep.get_map("Slope Degrees", (xmin, ymin, xmax, ymax), resolution=300)
        slope = _coerce_raster_like(slope)
    except Exception as e:
        print(f"Warning: slope download failed: {e}")

    return dem, slope


def save_raster(data_array: Any, path: Path) -> Path | None:
    data_array = _coerce_raster_like(data_array)
    if data_array is None:
        return None
    data_array.rio.to_raster(path)
    return path


def run_acquisition() -> None:
    claims = fetch_fema_claims()
    save_fema_claims(claims)

    tracts = load_nc_tracts()
    save_nc_tracts(tracts)

    acs = fetch_acs_tract_data()
    save_acs(acs)

    impervious = fetch_nlcd_impervious()
    if impervious is not None:
        save_raster(impervious, RAW_DIR / "nc_impervious_2021.tif")
    else:
        print("Skipping impervious save because download failed.")

    dem, slope = fetch_dem_and_slope()
    if dem is not None:
        save_raster(dem, RAW_DIR / "nc_dem_300m.tif")
    else:
        print("Skipping DEM save because DEM download failed.")

    if slope is not None:
        save_raster(slope, RAW_DIR / "nc_slope_300m.tif")
    else:
        print("Skipping slope save because slope download failed.")
