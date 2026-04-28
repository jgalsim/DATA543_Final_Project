from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

from src.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR

LOSS_COLUMNS = [
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "amountPaidOnIncreasedCostOfComplianceClaim",
    "netBuildingPaymentAmount",
    "netContentsPaymentAmount",
]


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    claims = gpd.read_parquet(RAW_DIR / "nc_fema_claims.parquet")
    tracts = gpd.read_parquet(RAW_DIR / "nc_tracts.parquet")
    acs = pd.read_parquet(RAW_DIR / "nc_acs_tracts.parquet")
    return claims, tracts, acs


def aggregate_claims_to_tracts(claims: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame) -> pd.DataFrame:
    joined = gpd.sjoin(
        claims.to_crs("EPSG:4326"),
        tracts[["GEOID", "geometry"]].to_crs("EPSG:4326"),
        how="inner",
        predicate="within",
    )

    aggregations = {"claim_count": ("GEOID", "size")}
    for col in LOSS_COLUMNS:
        if col in joined.columns:
            aggregations[f"sum_{col}"] = (col, "sum")
            aggregations[f"mean_{col}"] = (col, "mean")

    claims_by_tract = joined.groupby("GEOID").agg(**aggregations).reset_index()
    return claims_by_tract


def _empty_zonal(gdf: gpd.GeoDataFrame, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame({"GEOID": gdf["GEOID"].values})
    for col in ["mean", "median", "max", "min"]:
        out[f"{prefix}_{col}"] = np.nan
    return out


def _zonal_mean(gdf: gpd.GeoDataFrame, raster_path: Path, prefix: str) -> pd.DataFrame:
    if not raster_path.exists():
        return _empty_zonal(gdf, prefix)

    with rasterio.open(raster_path) as src:
        tract_proj = gdf.to_crs(src.crs)
        stats = zonal_stats(
            tract_proj.geometry,
            src.read(1),
            affine=src.transform,
            stats=["mean", "median", "max", "min"],
            nodata=src.nodata,
        )

    stats_df = pd.DataFrame(stats)
    stats_df.columns = [f"{prefix}_{col}" for col in stats_df.columns]
    stats_df.insert(0, "GEOID", gdf["GEOID"].values)
    return stats_df


def engineer_acs_features(acs: pd.DataFrame) -> pd.DataFrame:
    df = acs.copy()
    elderly_cols = [
        "age_65_66",
        "age_67_69",
        "age_70_74",
        "age_75_79",
        "age_80_84",
        "age_85_plus_m",
        "age_65_66_f",
        "age_67_69_f",
        "age_70_74_f",
        "age_75_79_f",
        "age_80_84_f",
        "age_85_plus_f",
    ]
    df["elderly_population"] = df[elderly_cols].sum(axis=1, min_count=1)
    df["pct_elderly"] = np.where(df["population"] > 0, df["elderly_population"] / df["population"], np.nan)

    occupied_units = df["owner_occupied"].fillna(0) + df["renter_occupied"].fillna(0)
    df["pct_renter"] = np.where(occupied_units > 0, df["renter_occupied"] / occupied_units, np.nan)
    df["pct_no_vehicle"] = np.where(
        df["total_households_vehicle"] > 0,
        df["no_vehicle"] / df["total_households_vehicle"],
        np.nan,
    )

    keep_cols = [
        "GEOID",
        "population",
        "housing_units",
        "median_income",
        "pct_renter",
        "pct_elderly",
        "pct_no_vehicle",
    ]
    return df[keep_cols]


def build_model_table() -> gpd.GeoDataFrame:
    claims, tracts, acs = load_inputs()

    claims_by_tract = aggregate_claims_to_tracts(claims, tracts)
    acs_features = engineer_acs_features(acs)
    impervious = _zonal_mean(tracts, RAW_DIR / "nc_impervious_2021.tif", "impervious")
    dem = _zonal_mean(tracts, RAW_DIR / "nc_dem_300m.tif", "elev")
    slope = _zonal_mean(tracts, RAW_DIR / "nc_slope_300m.tif", "slope")

    model_gdf = tracts.merge(claims_by_tract, on="GEOID", how="left")
    model_gdf = model_gdf.merge(acs_features, on="GEOID", how="left")
    model_gdf = model_gdf.merge(impervious, on="GEOID", how="left")
    model_gdf = model_gdf.merge(dem, on="GEOID", how="left")
    model_gdf = model_gdf.merge(slope, on="GEOID", how="left")

    model_gdf["claim_count"] = model_gdf["claim_count"].fillna(0).astype(int)
    model_gdf["has_claim"] = (model_gdf["claim_count"] > 0).astype(int)

    total_loss_cols = [c for c in model_gdf.columns if c.startswith("sum_")]
    if total_loss_cols:
        model_gdf["total_claim_payment"] = model_gdf[total_loss_cols].sum(axis=1, skipna=True)
    else:
        model_gdf["total_claim_payment"] = np.nan

    model_gdf["claim_rate_per_1000_units"] = np.where(
        model_gdf["housing_units"] > 0,
        (model_gdf["claim_count"] / model_gdf["housing_units"]) * 1000,
        np.nan,
    )
    model_gdf["claim_rate_per_1000_pop"] = np.where(
        model_gdf["population"] > 0,
        (model_gdf["claim_count"] / model_gdf["population"]) * 1000,
        np.nan,
    )

    model_gdf.to_parquet(PROCESSED_DIR / "nc_flood_model_table.parquet", index=False)
    model_gdf.drop(columns="geometry").to_csv(PROCESSED_DIR / "nc_flood_model_table.csv", index=False)
    return model_gdf


def make_quick_maps(model_gdf: gpd.GeoDataFrame) -> None:
    plot_specs = [
        ("claim_count", "North Carolina NFIP Claims by Tract", "viridis", "map_claim_count.png"),
        (
            "claim_rate_per_1000_units",
            "NFIP Claim Rate per 1,000 Housing Units",
            "plasma",
            "map_claim_rate_per_1000_units.png",
        ),
        ("impervious_mean", "Mean Impervious Surface by Tract", "magma", "map_impervious_mean.png"),
    ]

    for column, title, cmap, filename in plot_specs:
        if column not in model_gdf.columns:
            continue
        if model_gdf[column].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 7))
        model_gdf.plot(column=column, ax=ax, legend=True, cmap=cmap, linewidth=0.1, edgecolor="white")
        ax.set_title(title)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=300)
        plt.close(fig)
