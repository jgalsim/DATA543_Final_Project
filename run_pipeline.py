from __future__ import annotations

import json

from src.acquire import run_acquisition
from src.features import build_model_table, make_quick_maps
from src.model import run_baseline_models, run_impervious_scenario


def main() -> None:
    print("Step 1: Downloading and saving raw data...")
    run_acquisition()

    print("Step 2: Building tract-level model table...")
    model_gdf = build_model_table()

    print("Step 3: Creating quick maps...")
    make_quick_maps(model_gdf)

    print("Step 4: Running baseline models...")
    model_df = model_gdf.drop(columns="geometry")
    results = run_baseline_models(model_df)
    print(
        json.dumps(
            {
                "regression_metrics": results.regression_metrics,
                "classification_metrics": results.classification_metrics,
                "feature_importance": results.feature_importance,
            },
            indent=2,
        )
    )

    print("Step 5: Running imperviousness reduction scenario...")
    try:
        scenario = run_impervious_scenario(model_df, reduction_fraction=0.10)
        print(scenario.head())
    except Exception as e:
        print(f"Scenario step skipped: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
