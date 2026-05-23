"""Mock flow test for the energy-forecasting LLM-FE adapter.

Run from the LLM-FE repository root after placing energy_forecasting_adapter.py:

    python tests/mock_energy_forecasting_flow.py

This does not call a real LLM.  It executes two mock candidate bodies: one valid
row-wise feature and one invalid global-statistic feature that should be rejected
by the row-wise guard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from energy_forecasting_adapter import (  # noqa: E402
    CandidatePolicyConfig,
    EnergyForecastingConfig,
    LightGBMSearchConfig,
    TemporalSplitConfig,
    candidate_fn_from_body,
    score_candidate_feature,
)


def make_mock_energy_data(n_days: int = 320, buildings: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    ts = pd.date_range("2016-01-01", periods=n_days * 24, freq="h")
    frames = []
    rng = np.random.default_rng(155)
    for b in range(buildings):
        df = pd.DataFrame({"timestamp": ts})
        hour = df["timestamp"].dt.hour
        doy = df["timestamp"].dt.dayofyear
        df["building_id"] = b
        df["site_id"] = b % 2
        df["meter"] = pd.Series(np.full(len(df), b % 4), dtype="category")
        df["primary_use"] = pd.Series(["Education" if b % 2 == 0 else "Office"] * len(df), dtype="category")
        df["square_feet"] = 20000 + b * 15000
        df["floor_count"] = 2 + b
        df["air_temperature"] = 18 + 8 * np.sin(2 * np.pi * (doy / 365.0)) + rng.normal(0, 1.0, len(df))
        df["cloud_coverage"] = np.clip(4 + rng.normal(0, 1.0, len(df)), 0, 8)
        df["dew_temperature"] = df["air_temperature"] - 4 + rng.normal(0, 0.5, len(df))
        df["precip_depth_1_hr"] = np.maximum(0, rng.normal(0.2, 0.5, len(df)))
        df["sea_level_pressure"] = 1012 + rng.normal(0, 3, len(df))
        df["wind_direction"] = rng.uniform(0, 360, len(df))
        df["wind_speed"] = np.maximum(0, rng.normal(3, 1, len(df)))
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["dayofweek"] = df["timestamp"].dt.dayofweek.astype("category")
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
        df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype("uint8")
        df["is_holiday_any"] = 0
        df["is_business_hours"] = hour.between(8, 18).astype("uint8")
        df["CDH_18C"] = np.maximum(0, df["air_temperature"] - 18)
        df["HDH_18C"] = np.maximum(0, 18 - df["air_temperature"])
        df["is_hot_24C"] = (df["air_temperature"] > 24).astype("uint8")
        df["is_cold_10C"] = (df["air_temperature"] < 10).astype("uint8")
        df["dewpoint_depression"] = df["air_temperature"] - df["dew_temperature"]
        df["log_sqft"] = np.log1p(df["square_feet"])
        df["year_built_clipped"] = 1990 + b
        raw_target = (
            50
            + 0.002 * df["square_feet"]
            + 2.0 * df["CDH_18C"]
            + 1.5 * df["HDH_18C"]
            + 10.0 * df["is_business_hours"]
            + rng.normal(0, 3, len(df))
        )
        df["meter_reading"] = np.maximum(0, raw_target)
        frames.append(df)
    full = pd.concat(frames, axis=0).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    outputs = full.pop("meter_reading")
    return full, outputs


def main() -> None:
    inputs, outputs = make_mock_energy_data()
    cfg = EnergyForecastingConfig(
        split=TemporalSplitConfig(
            train_start_day=0,
            train_end_day_exclusive=280,
            inner_eval_start_day=280,
            inner_eval_end_day_exclusive=320,
            final_test_start_day=320,
            final_test_end_day_exclusive=366,
            early_stopping_tail_days=7,
        ),
        lgbm=LightGBMSearchConfig(
            n_estimators=40,
            early_stopping_rounds=5,
            early_stopping_min_delta=1e-5,
            min_data_in_leaf=20,
            log_period=0,
            seed=155,
            num_threads=1,
        ),
        candidate_policy=CandidatePolicyConfig(max_new_columns=8, rowwise_probe_rows=8, rowwise_probe_seed=155),
    )

    valid_body = """
out = pd.DataFrame(index=df_input.index)
out['llmfe_temp_x_log_sqft'] = df_input['air_temperature'] * df_input['log_sqft']
out['llmfe_business_hot'] = df_input['is_business_hours'].astype(float) * df_input['CDH_18C']
return out
""".strip()

    invalid_body = """
out = pd.DataFrame(index=df_input.index)
out['llmfe_global_temp_centered'] = df_input['air_temperature'] - df_input['air_temperature'].mean()
return out
""".strip()

    valid_result = score_candidate_feature(inputs, outputs, candidate_fn_from_body(valid_body), cfg)
    print("VALID STATUS:", valid_result.candidate_status)
    print("VALID SCORE:", valid_result.score_for_llmfe)
    print("VALID COLUMNS:", valid_result.candidate_columns)
    assert valid_result.candidate_status == "valid", valid_result.details
    assert valid_result.n_candidate_columns == 2

    invalid_result = score_candidate_feature(inputs, outputs, candidate_fn_from_body(invalid_body), cfg)
    print("INVALID STATUS:", invalid_result.candidate_status)
    print("INVALID REASON:", invalid_result.details.get("reason"))
    assert invalid_result.candidate_status == "invalid"
    assert "row-wise" in invalid_result.details.get("reason", "") or "stable" in invalid_result.details.get("reason", "")

    print("Mock flow passed.")


if __name__ == "__main__":
    main()
