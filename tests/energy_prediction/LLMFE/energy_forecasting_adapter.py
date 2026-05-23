"""Energy-forecasting evaluator adapter for LLM-FE.

Place this file at the root of the LLM-FE repository.

The adapter intentionally keeps the LLM-FE search machinery unchanged.  It
replaces only the task-specific evaluator: a generated ``modify_features``
function is treated as a row-wise/on-timestamp supplementary feature candidate,
concatenated with the fixed baseline feature set, and scored by a temporal
train/inner-eval LightGBM protocol.

Search protocol used here:
    days 0-279   : model-training region
    last N days  : early-stopping tail within the training region
    days 280-319 : inner evaluation region for LLM-FE candidate scoring

The final test region, days 320-365, must not be loaded into the LLM-FE search
process.  Final frozen-feature evaluation should be run separately after the
candidate has been selected.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from flask import config
import numpy as np
import pandas as pd
import inspect

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover - handled at runtime for clearer error messages.
    lgb = None  # type: ignore[assignment]
    print(f"Warning: lightgbm import failed with error: {exc!r}. Candidate scoring will not work.", flush=True)
    _LIGHTGBM_IMPORT_ERROR = exc
else:
    _LIGHTGBM_IMPORT_ERROR = None


CandidateFn = Callable[[pd.DataFrame], pd.DataFrame]


RAW_BASE_FEATURES: List[str] = [
    "meter",
    "primary_use",
    "square_feet",
    "year_built",
    "floor_count",
    # "timestamp",
    # "timestamp_gmt",
    "time_diff_hours",
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
]

ENGINEERED_BASE_FEATURES: List[str] = [
    "hour_sin",
    "hour_cos",
    "dayofweek",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_holiday_any",
    "is_na_holiday",
    "is_eu_holiday",
    "is_business_hours",
    "CDH_18C",
    "HDH_18C",
    "is_hot_24C",
    "is_cold_10C",
    "dewpoint_depression",
    "log_sqft",
    "year_built_clipped",
]

BASE_CATEGORICAL_COLUMNS: List[str] = ["primary_use", "meter", "dayofweek"]

FORBIDDEN_COLUMNS: List[str] = [
    "building_id",
    "site_id",
    "meter_reading",
    "pair_id",
    "ts_idx",
]


@dataclass(frozen=True)
class LightGBMSearchConfig:
    """LightGBM configuration used during candidate scoring."""

    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_data_in_leaf: int = 500
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    max_bin: int = 255
    n_estimators: int = 12_000
    early_stopping_rounds: int = 200
    early_stopping_min_delta: float = 10e-4
    log_period: int = 800
    seed: int = 155
    num_threads: int = 0
    device_type: str = "cpu"


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Day-index split for the LLM-FE search stage."""

    train_start_day: int = 0
    train_end_day_exclusive: int = 280
    inner_eval_start_day: int = 280
    inner_eval_end_day_exclusive: int = 320
    final_test_start_day: int = 320
    final_test_end_day_exclusive: int = 366
    early_stopping_tail_days: int = 14


@dataclass(frozen=True)
class CandidatePolicyConfig:
    """Safety and scope policy for generated feature candidates."""

    max_new_columns: int = 32
    rowwise_probe_rows: int = 12
    rowwise_probe_seed: int = 155
    allow_object_as_category: bool = True
    reject_candidate_input_mutation: bool = True
    require_nonempty_features: bool = False
    max_abs_numeric_value: float = 1.0e12


@dataclass(frozen=True)
class EnergyForecastingConfig:
    """Full adapter configuration.

    ``out_dir`` is optional.  If provided, the adapter writes candidate audit
    records to ``<out_dir>/candidate_audit.jsonl``.
    """

    label_col: str = "meter_reading"
    timestamp_col: str = "timestamp"
    use_log1p_target: bool = True
    baseline_features: Tuple[str, ...] = tuple(RAW_BASE_FEATURES + ENGINEERED_BASE_FEATURES)
    base_categorical_cols: Tuple[str, ...] = tuple(BASE_CATEGORICAL_COLUMNS)
    candidate_input_extra_cols: Tuple[str, ...] = ("timestamp",)
    forbidden_output_cols: Tuple[str, ...] = tuple(FORBIDDEN_COLUMNS + ["meter_reading"])
    split: TemporalSplitConfig = field(default_factory=TemporalSplitConfig)
    lgbm: LightGBMSearchConfig = field(default_factory=LightGBMSearchConfig)
    candidate_policy: CandidatePolicyConfig = field(default_factory=CandidatePolicyConfig)
    out_dir: Optional[str] = None
    prompt_sample_rows: int = 10

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "EnergyForecastingConfig":
        if not payload:
            return cls()

        def _nested(dc_cls, value):
            if value is None:
                return dc_cls()
            if isinstance(value, dc_cls):
                return value
            return dc_cls(**dict(value))

        data = dict(payload)
        split = _nested(TemporalSplitConfig, data.pop("split", None))
        lgbm_cfg = _nested(LightGBMSearchConfig, data.pop("lgbm", None))
        policy = _nested(CandidatePolicyConfig, data.pop("candidate_policy", None))
        return cls(split=split, lgbm=lgbm_cfg, candidate_policy=policy, **data)

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateScoreResult:
    score_for_llmfe: float
    inner_eval_rmse_log1p: float
    train_rmse_log1p: float
    best_iteration: int
    candidate_status: str
    runtime_seconds: float
    n_candidate_columns: int
    candidate_columns: List[str]
    prompt_inputs: pd.DataFrame
    prompt_outputs: pd.Series
    details: Dict[str, Any] = field(default_factory=dict)
    candidate_id: Optional[str] = None
    candidate_source: Optional[str] = None
    is_best_so_far: bool = False


def _trace(config: EnergyForecastingConfig, message: str) -> None:
    """Write a minimal live runtime trace to stdout and runtime_trace.log."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[EnergyLLMFE {stamp}] {message}"

    print(line, flush=True)

    if config.out_dir:
        out_dir = Path(config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "runtime_trace.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def choose_baseline_feature_columns(all_cols: Iterable[str], configured: Sequence[str]) -> List[str]:
    """Return baseline columns present in the dataset, preserving configured order."""
    present = set(all_cols)
    return [c for c in configured if c in present]


def load_parquet_dataset(path: str | Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet path not found: {path}")
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def build_day_index(df: pd.DataFrame, timestamp_col: str) -> pd.Series:
    if timestamp_col not in df.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_col}")
    ts = pd.to_datetime(df[timestamp_col], utc=False)
    if ts.isna().any():
        raise ValueError(f"{timestamp_col} contains NaT values; cannot build temporal split.")
    day_floor = ts.dt.floor("D")
    base_day = day_floor.min()
    return (day_floor - base_day).dt.days.astype(int)


def load_development_data(
    data_parquet: str | Path,
    config: EnergyForecastingConfig,
    *,
    physically_exclude_final_test: bool = True,
) -> Dict[str, Any]:
    """Load only the development region used by the LLM-FE search.

    The returned ``inputs`` never include the label column.  ``outputs`` is the
    raw target, aligned row-wise with ``inputs``.
    """
    required_for_split = [config.label_col, config.timestamp_col]
    probe = load_parquet_dataset(data_parquet, columns=None)
    if config.timestamp_col not in probe.columns or config.label_col not in probe.columns:
        raise KeyError(f"Dataset must contain {config.timestamp_col!r} and {config.label_col!r}.")

    probe = probe.sort_values(config.timestamp_col, kind="mergesort").reset_index(drop=True)
    day_index = build_day_index(probe, config.timestamp_col)

    if physically_exclude_final_test:
        dev_mask = day_index < config.split.final_test_start_day
        if not bool(dev_mask.any()):
            raise ValueError("Development split is empty before final_test_start_day.")
        probe = probe.loc[dev_mask].reset_index(drop=True)
        day_index = build_day_index(probe, config.timestamp_col)

    if int(day_index.max()) >= config.split.final_test_start_day:
        raise ValueError(
            "Final-test rows are present in the LLM-FE search dataframe. "
            "Physically exclude the final future slice before search."
        )

    outputs = probe[config.label_col].copy()
    inputs = probe.drop(columns=[config.label_col]).copy()
    return {
        "inputs": inputs,
        "outputs": outputs,
        "config": config.to_jsonable(),
    }


def metadata_for_energy_features() -> Dict[str, str]:
    """Feature descriptions used by LLM-FE prompt construction."""
    return {
        "meter": "Meter type identifier: 0 electricity, 1 chilled water, 2 steam, 3 hot water.",
        "primary_use": "Building primary-use category such as Education or Office.",
        "square_feet": "Gross building floor area in square feet.",
        "floor_count": "Number of building floors.",
        "air_temperature": "Outdoor air temperature in degrees Celsius for the matched site-hour.",
        "cloud_coverage": "Sky cloud cover in oktas, typically 0 to 8.",
        "dew_temperature": "Outdoor dew-point temperature in degrees Celsius.",
        "precip_depth_1_hr": "One-hour precipitation depth in millimeters.",
        "sea_level_pressure": "Sea-level atmospheric pressure in millibar/hectopascals.",
        "wind_direction": "Wind bearing in degrees on a 0 to 360 compass scale.",
        "wind_speed": "Wind speed in meters per second.",
        "hour_sin": "Sine encoding of hour of day.",
        "hour_cos": "Cosine encoding of hour of day.",
        "dayofweek": "Day of week code with Monday=0.",
        "doy_sin": "Sine encoding of day of year.",
        "doy_cos": "Cosine encoding of day of year.",
        "is_weekend": "Weekend indicator.",
        "is_holiday_any": "Holiday indicator.",
        "is_business_hours": "Business-hours indicator.",
        "CDH_18C": "Cooling degree hours relative to 18 Celsius.",
        "HDH_18C": "Heating degree hours relative to 18 Celsius.",
        "is_hot_24C": "Indicator for hot weather above 24 Celsius.",
        "is_cold_10C": "Indicator for cold weather below 10 Celsius.",
        "dewpoint_depression": "Difference between air temperature and dew temperature.",
        "log_sqft": "Log-transformed square-foot area.",
        "year_built_clipped": "Clipped construction/opening year feature.",
        "timestamp": "Hourly observation timestamp; may be used only for row-wise on-timestamp transformations.",
    }


def make_lgbm_params(config: EnergyForecastingConfig) -> Dict[str, Any]:
    cfg = config.lgbm
    params: Dict[str, Any] = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "max_depth": cfg.max_depth,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "lambda_l2": cfg.lambda_l2,
        "max_bin": cfg.max_bin,
        "seed": cfg.seed,
        "num_threads": cfg.num_threads,
        "verbosity": -1,
    }
    if cfg.device_type in {"gpu", "cuda"}:
        params["device_type"] = cfg.device_type
    return params


def prepare_tabular_matrices(
    df: pd.DataFrame,
    label_col: str,
    drop_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Local equivalent of the project's tree_prepare_tabular_matrices."""
    if label_col not in df.columns:
        raise KeyError(f"Missing label_col: {label_col}")
    y = df[label_col].to_numpy()
    X = df.drop(columns=[label_col] + [c for c in drop_cols if c in df.columns], errors="ignore").copy()
    cat_present = [c for c in categorical_cols if c in X.columns]
    for c in cat_present:
        if not isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    if obj_cols:
        raise TypeError(f"Object dtype columns found: {obj_cols}. Convert them to category or numeric.")
    return X, y, cat_present


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _safe_worst_score(reason: str, candidate_source: Optional[str] = None) -> CandidateScoreResult:
    return CandidateScoreResult(
        score_for_llmfe=-1.0e12,
        inner_eval_rmse_log1p=float("inf"),
        train_rmse_log1p=float("inf"),
        best_iteration=0,
        candidate_status="invalid",
        runtime_seconds=0.0,
        n_candidate_columns=0,
        candidate_columns=[],
        prompt_inputs=pd.DataFrame(),
        prompt_outputs=pd.Series(dtype=float),
        details={"reason": reason},
        candidate_source=candidate_source,
    )


def _candidate_input_columns(inputs: pd.DataFrame, config: EnergyForecastingConfig) -> List[str]:
    baseline_cols = choose_baseline_feature_columns(inputs.columns, config.baseline_features)
    extra_cols = [c for c in config.candidate_input_extra_cols if c in inputs.columns]
    return list(dict.fromkeys(extra_cols + baseline_cols))


def _normalize_candidate_output(
    feature_df: Any,
    *,
    expected_rows: int,
    config: EnergyForecastingConfig,
    base_input_cols: Sequence[str],
) -> pd.DataFrame:
    if feature_df is None:
        feature_df = pd.DataFrame(index=range(expected_rows))
    if not isinstance(feature_df, pd.DataFrame):
        feature_df = pd.DataFrame(feature_df)
    if len(feature_df) != expected_rows:
        raise ValueError(f"Candidate row mismatch: {len(feature_df)} vs {expected_rows}")

    feature_df = feature_df.copy().reset_index(drop=True)
    feature_df.columns = [str(c) for c in feature_df.columns]
    if len(set(feature_df.columns)) != len(feature_df.columns):
        raise ValueError("Candidate returned duplicate feature names.")

    forbidden = set(config.forbidden_output_cols) | set(base_input_cols)
    collisions = sorted(c for c in feature_df.columns if c in forbidden)
    if collisions:
        raise ValueError(f"Candidate returned forbidden or existing columns: {collisions[:20]}")

    if config.candidate_policy.require_nonempty_features and feature_df.shape[1] == 0:
        raise ValueError("Candidate returned no features.")
    if feature_df.shape[1] > config.candidate_policy.max_new_columns:
        raise ValueError(
            f"Candidate returned {feature_df.shape[1]} columns, exceeding max_new_columns="
            f"{config.candidate_policy.max_new_columns}."
        )

    # LightGBM can handle category dtype; object must be converted or rejected.
    for col in list(feature_df.columns):
        s = feature_df[col]
        if pd.api.types.is_bool_dtype(s):
            feature_df[col] = s.astype("uint8")
        elif pd.api.types.is_object_dtype(s):
            if config.candidate_policy.allow_object_as_category:
                feature_df[col] = s.astype("category")
            else:
                raise TypeError(f"Candidate column {col!r} has object dtype.")
        elif pd.api.types.is_datetime64_any_dtype(s):
            raise TypeError(f"Candidate column {col!r} is datetime; convert to numeric/categorical explicitly.")

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        numeric = feature_df[numeric_cols].to_numpy(dtype=float, copy=False)
        if not np.isfinite(numeric).all():
            raise ValueError("Candidate produced non-finite numeric values.")
        max_abs = float(np.nanmax(np.abs(numeric))) if numeric.size else 0.0
        if max_abs > config.candidate_policy.max_abs_numeric_value:
            raise ValueError(
                f"Candidate numeric magnitude {max_abs:g} exceeds max_abs_numeric_value="
                f"{config.candidate_policy.max_abs_numeric_value:g}."
            )
    return feature_df


def _call_candidate(
    candidate_fn: CandidateFn,
    candidate_input: pd.DataFrame,
    *,
    config: EnergyForecastingConfig,
) -> pd.DataFrame:
    input_before = candidate_input.copy(deep=True)
    raw_out = candidate_fn(candidate_input.copy(deep=True))
    if config.candidate_policy.reject_candidate_input_mutation:
        try:
            pd.testing.assert_frame_equal(input_before, candidate_input, check_dtype=False, check_exact=False)
        except AssertionError as exc:
            raise ValueError("Candidate mutated its input dataframe in-place.") from exc
    return _normalize_candidate_output(
        raw_out,
        expected_rows=len(candidate_input),
        config=config,
        base_input_cols=candidate_input.columns,
    )


def _series_equal(a: pd.Series, b: pd.Series) -> bool:
    if len(a) != len(b):
        return False
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        av = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
        bv = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
        same_nan = np.isnan(av) == np.isnan(bv)
        close = np.isclose(np.nan_to_num(av), np.nan_to_num(bv), rtol=1e-8, atol=1e-10)
        return bool(np.all(same_nan & close))
    return bool((a.astype(str).fillna("<NA>").to_numpy() == b.astype(str).fillna("<NA>").to_numpy()).all())


def assert_rowwise_candidate(
    candidate_fn: CandidateFn,
    candidate_input: pd.DataFrame,
    config: EnergyForecastingConfig,
) -> None:
    """Black-box guard for row-wise/on-timestamp feature candidates.

    The same rows are evaluated once as a small batch and once as individual
    rows.  Features depending on full-window statistics, groupby transforms,
    rolling windows, future rows, or sample-level ranks typically fail this
    stability check.
    """
    n_rows = len(candidate_input)
    if n_rows == 0:
        raise ValueError("Cannot validate row-wise policy on empty candidate input.")
    probe_n = min(config.candidate_policy.rowwise_probe_rows, n_rows)
    rng = np.random.default_rng(config.candidate_policy.rowwise_probe_seed)
    positions = np.sort(rng.choice(n_rows, size=probe_n, replace=False))
    probe = candidate_input.iloc[positions].reset_index(drop=True)

    batch_out = _call_candidate(candidate_fn, probe, config=config)
    for i in range(probe_n):
        single = probe.iloc[[i]].reset_index(drop=True)
        single_out = _call_candidate(candidate_fn, single, config=config)
        if list(batch_out.columns) != list(single_out.columns):
            raise ValueError("Candidate is not row-wise stable: output columns differ for single-row calls.")
        for col in batch_out.columns:
            if not _series_equal(
                batch_out.loc[[i], col].reset_index(drop=True), single_out[col].reset_index(drop=True)
            ):
                raise ValueError(
                    "Candidate is not row-wise/on-timestamp stable; feature "
                    f"{col!r} changes when evaluated independently per row."
                )


def _apply_candidate_to_split(
    candidate_fn: CandidateFn,
    inputs: pd.DataFrame,
    mask: np.ndarray,
    config: EnergyForecastingConfig,
    *,
    validate_rowwise: bool,
) -> pd.DataFrame:
    input_cols = _candidate_input_columns(inputs, config)
    candidate_input = inputs.loc[mask, input_cols].reset_index(drop=True)
    if validate_rowwise:
        assert_rowwise_candidate(candidate_fn, candidate_input, config)
    return _call_candidate(candidate_fn, candidate_input, config=config)


def _build_model_frame(
    inputs: pd.DataFrame,
    outputs: pd.Series | np.ndarray,
    mask: np.ndarray,
    candidate_df: pd.DataFrame,
    config: EnergyForecastingConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    base_cols = choose_baseline_feature_columns(inputs.columns, config.baseline_features)
    if not base_cols:
        raise ValueError("No baseline feature columns were found in the input dataframe.")
    base = inputs.loc[mask, base_cols].reset_index(drop=True).copy()
    for c in config.base_categorical_cols:
        if c in base.columns and not isinstance(base[c].dtype, pd.CategoricalDtype):
            base[c] = base[c].astype("category")
    candidate = candidate_df.reset_index(drop=True).copy()
    candidate_cat_cols: List[str] = []
    for c in candidate.columns:
        if isinstance(candidate[c].dtype, pd.CategoricalDtype):
            candidate_cat_cols.append(c)
    y = pd.Series(outputs).loc[mask].reset_index(drop=True)
    if config.use_log1p_target:
        y = np.log1p(y.to_numpy(dtype=float)).astype(np.float32)
    else:
        y = y.to_numpy(dtype=np.float32)
    model_df = pd.concat([base, candidate], axis=1)
    model_df[config.label_col] = y
    categorical_cols = list(
        dict.fromkeys([c for c in config.base_categorical_cols if c in model_df.columns] + candidate_cat_cols)
    )
    return model_df, categorical_cols


def _train_lgbm_score(
    inputs: pd.DataFrame,
    outputs: pd.Series | np.ndarray,
    candidate_fn: CandidateFn,
    config: EnergyForecastingConfig,
) -> CandidateScoreResult:
    if lgb is None:
        raise ImportError("lightgbm is required for candidate scoring.") from _LIGHTGBM_IMPORT_ERROR

    set_seed(config.lgbm.seed)
    day_index = build_day_index(inputs, config.timestamp_col)
    split = config.split

    if int(day_index.max()) >= split.final_test_start_day:
        raise ValueError("LLM-FE search input contains final-test rows; aborting candidate evaluation.")

    train_mask = ((day_index >= split.train_start_day) & (day_index < split.train_end_day_exclusive)).to_numpy()
    inner_eval_mask = (
        (day_index >= split.inner_eval_start_day) & (day_index < split.inner_eval_end_day_exclusive)
    ).to_numpy()
    tail_start = split.train_end_day_exclusive - split.early_stopping_tail_days
    inner_fit_mask = ((day_index >= split.train_start_day) & (day_index < tail_start)).to_numpy()
    inner_tail_mask = ((day_index >= tail_start) & (day_index < split.train_end_day_exclusive)).to_numpy()

    if not train_mask.any() or not inner_eval_mask.any() or not inner_fit_mask.any() or not inner_tail_mask.any():
        raise ValueError(
            "Temporal split produced an empty segment. Check train/eval day boundaries against the dataset."
        )
    _trace(
        config,
        "temporal split rows "
        f"inner_fit={int(inner_fit_mask.sum())} "
        f"inner_tail={int(inner_tail_mask.sum())} "
        f"train={int(train_mask.sum())} "
        f"inner_eval={int(inner_eval_mask.sum())}",
    )

    _trace(config, "applying candidate features")

    candidate_inner_fit = _apply_candidate_to_split(candidate_fn, inputs, inner_fit_mask, config, validate_rowwise=True)
    candidate_inner_tail = _apply_candidate_to_split(
        candidate_fn, inputs, inner_tail_mask, config, validate_rowwise=False
    )
    candidate_train = _apply_candidate_to_split(candidate_fn, inputs, train_mask, config, validate_rowwise=False)
    candidate_eval = _apply_candidate_to_split(candidate_fn, inputs, inner_eval_mask, config, validate_rowwise=False)
    candidate_cols_preview = list(candidate_train.columns)
    _trace(
        config,
        "candidate features accepted " f"n_cols={len(candidate_cols_preview)} " f"cols={candidate_cols_preview[:12]}",
    )

    inner_fit_df, inner_fit_cats = _build_model_frame(inputs, outputs, inner_fit_mask, candidate_inner_fit, config)
    inner_tail_df, inner_tail_cats = _build_model_frame(inputs, outputs, inner_tail_mask, candidate_inner_tail, config)
    train_df, train_cats = _build_model_frame(inputs, outputs, train_mask, candidate_train, config)
    eval_df, eval_cats = _build_model_frame(inputs, outputs, inner_eval_mask, candidate_eval, config)

    cat_cols = list(dict.fromkeys(inner_fit_cats + inner_tail_cats + train_cats + eval_cats))
    X_inner_fit, y_inner_fit, cat_present_inner = prepare_tabular_matrices(
        inner_fit_df, config.label_col, drop_cols=[], categorical_cols=cat_cols
    )
    X_inner_tail, y_inner_tail, _ = prepare_tabular_matrices(
        inner_tail_df, config.label_col, drop_cols=[], categorical_cols=cat_cols
    )
    X_train, y_train, cat_present_train = prepare_tabular_matrices(
        train_df, config.label_col, drop_cols=[], categorical_cols=cat_cols
    )
    X_eval, y_eval, _ = prepare_tabular_matrices(eval_df, config.label_col, drop_cols=[], categorical_cols=cat_cols)
    _trace(
        config,
        "matrix shapes "
        f"X_inner_fit={X_inner_fit.shape} "
        f"X_inner_tail={X_inner_tail.shape} "
        f"X_train={X_train.shape} "
        f"X_eval={X_eval.shape}",
    )

    params = make_lgbm_params(config)
    dtrain_inner = lgb.Dataset(
        X_inner_fit,
        label=y_inner_fit,
        categorical_feature=cat_present_inner,
        free_raw_data=False,
    )
    dval_inner = lgb.Dataset(
        X_inner_tail,
        label=y_inner_tail,
        categorical_feature=cat_present_inner,
        reference=dtrain_inner,
        free_raw_data=False,
    )
    inner_evals_result: Dict[str, Any] = {}
    _trace(
        config,
        "stage1 early-stopping fit start "
        f"num_boost_round={config.lgbm.n_estimators} "
        f"log_period={config.lgbm.log_period}",
    )
    booster_inner = lgb.train(
        params=params,
        train_set=dtrain_inner,
        num_boost_round=config.lgbm.n_estimators,
        valid_sets=[dtrain_inner, dval_inner],
        valid_names=["train", "tail"],
        callbacks=[
            lgb.record_evaluation(inner_evals_result),
            lgb.log_evaluation(period=config.lgbm.log_period),
            lgb.early_stopping(
                stopping_rounds=config.lgbm.early_stopping_rounds,
                first_metric_only=True,
                verbose=True,
                min_delta=config.lgbm.early_stopping_min_delta,
            ),
        ],
    )
    best_iter = int(booster_inner.best_iteration or config.lgbm.n_estimators)
    _trace(config, f"stage1 done best_iter={best_iter}")

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_present_train,
        free_raw_data=False,
    )
    outer_evals_result: Dict[str, Any] = {}
    _trace(config, f"stage2 refit start num_boost_round={best_iter}")
    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=best_iter,
        valid_sets=[dtrain],
        valid_names=["train"],
        callbacks=[lgb.record_evaluation(outer_evals_result), lgb.log_evaluation(period=config.lgbm.log_period * 4)],
    )
    pred_eval = booster.predict(X_eval, num_iteration=best_iter)
    eval_rmse = _rmse(pred_eval, y_eval)
    train_rmse = float(outer_evals_result["train"]["rmse"][-1])
    _trace(
        config,
        "stage2 done "
        f"train_rmse={train_rmse:.6f} "
        f"inner_eval_rmse={eval_rmse:.6f} "
        f"score_for_llmfe={-eval_rmse:.6f}",
    )

    candidate_cols = list(candidate_train.columns)
    prompt_inputs, prompt_outputs = _make_prompt_sample(inputs, outputs, config)
    return CandidateScoreResult(
        score_for_llmfe=-eval_rmse,
        inner_eval_rmse_log1p=eval_rmse,
        train_rmse_log1p=train_rmse,
        best_iteration=best_iter,
        candidate_status="valid",
        runtime_seconds=0.0,
        n_candidate_columns=len(candidate_cols),
        candidate_columns=candidate_cols,
        prompt_inputs=prompt_inputs,
        prompt_outputs=prompt_outputs,
        details={
            "train_rows": int(train_mask.sum()),
            "inner_eval_rows": int(inner_eval_mask.sum()),
            "inner_fit_rows": int(inner_fit_mask.sum()),
            "inner_tail_rows": int(inner_tail_mask.sum()),
        },
    )


def _make_prompt_sample(
    inputs: pd.DataFrame,
    outputs: pd.Series | np.ndarray,
    config: EnergyForecastingConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    input_cols = _candidate_input_columns(inputs, config)
    day_index = build_day_index(inputs, config.timestamp_col)
    train_mask = (
        (day_index >= config.split.train_start_day) & (day_index < config.split.train_end_day_exclusive)
    ).to_numpy()
    train_inputs = inputs.loc[train_mask, input_cols].reset_index(drop=True)
    train_outputs = pd.Series(outputs).loc[train_mask].reset_index(drop=True)
    n = min(config.prompt_sample_rows, len(train_inputs))
    if n <= 0:
        return train_inputs.head(0), train_outputs.head(0)
    sample = train_inputs.sample(n=n, random_state=config.lgbm.seed).sort_index()
    sample_outputs = train_outputs.loc[sample.index].reset_index(drop=True)
    return sample.reset_index(drop=True), sample_outputs


def _candidate_source_from_fn(candidate_fn: CandidateFn) -> Optional[str]:
    """Best-effort fallback for local/mock candidates.

    For real LLM-FE runs, prefer the source injected by llmfe/evaluator.py into:
        data["_llmfe_candidate_function_source"]

    inspect.getsource can fail for dynamically compiled LLM functions.
    """
    try:
        return inspect.getsource(candidate_fn)
    except Exception:
        return None


def _next_candidate_id(out_dir: Path) -> str:
    audit_path = out_dir / "candidate_audit.jsonl"
    if not audit_path.exists():
        return "000001"
    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
    except OSError:
        n = 0
    return f"{n + 1:06d}"


def _safe_json_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _materialize_candidate_artifacts(
    config: EnergyForecastingConfig,
    result: CandidateScoreResult,
) -> None:
    """Persist source for completed candidates and update best-so-far snapshot."""
    if not config.out_dir:
        return

    out_dir = Path(config.out_dir)
    source = result.candidate_source
    if not source:
        return

    candidates_dir = out_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    candidate_id = result.candidate_id or _next_candidate_id(out_dir)
    result.candidate_id = candidate_id

    candidate_path = candidates_dir / f"candidate_{candidate_id}.py"
    if not candidate_path.exists():
        candidate_path.write_text(source.rstrip() + "\n", encoding="utf-8")

    if result.candidate_status != "valid":
        return

    best_json_path = out_dir / "best_candidate.json"
    best_py_path = out_dir / "best_candidate.py"

    best_payload = _safe_json_load(best_json_path)
    old_score = best_payload.get("score_for_llmfe")
    is_better = old_score is None or float(result.score_for_llmfe) > float(old_score)

    if not is_better:
        return

    result.is_best_so_far = True

    best_py_path.write_text(source.rstrip() + "\n", encoding="utf-8")

    best_payload = {
        "candidate_id": candidate_id,
        "score_for_llmfe": result.score_for_llmfe,
        "inner_eval_rmse_log1p": result.inner_eval_rmse_log1p,
        "train_rmse_log1p": result.train_rmse_log1p,
        "best_iteration": result.best_iteration,
        "candidate_status": result.candidate_status,
        "runtime_seconds": result.runtime_seconds,
        "n_candidate_columns": result.n_candidate_columns,
        "candidate_columns": result.candidate_columns,
        "candidate_file": str(candidate_path.relative_to(out_dir)),
        "best_candidate_file": "best_candidate.py",
        "updated_ts": time.time(),
        "details": result.details,
    }

    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2, default=str)


def _write_audit(config: EnergyForecastingConfig, result: CandidateScoreResult) -> None:
    if not config.out_dir:
        return

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.candidate_id is None:
        result.candidate_id = _next_candidate_id(out_dir)

    _materialize_candidate_artifacts(config, result)

    payload = {
        "ts": time.time(),
        "candidate_id": result.candidate_id,
        "score_for_llmfe": result.score_for_llmfe,
        "inner_eval_rmse_log1p": result.inner_eval_rmse_log1p,
        "train_rmse_log1p": result.train_rmse_log1p,
        "best_iteration": result.best_iteration,
        "candidate_status": result.candidate_status,
        "runtime_seconds": result.runtime_seconds,
        "n_candidate_columns": result.n_candidate_columns,
        "candidate_columns": result.candidate_columns,
        "candidate_source": result.candidate_source,
        "candidate_file": (f"candidates/candidate_{result.candidate_id}.py" if result.candidate_source else None),
        "is_best_so_far": result.is_best_so_far,
        "details": result.details,
    }

    with open(out_dir / "candidate_audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def score_candidate_feature(
    inputs: pd.DataFrame,
    outputs: pd.Series | np.ndarray,
    candidate_fn: CandidateFn,
    config: EnergyForecastingConfig | Mapping[str, Any] | None = None,
    *,
    candidate_source: Optional[str] = None,
) -> CandidateScoreResult:
    """Score one generated feature candidate by temporal LightGBM evaluation."""
    source = candidate_source or _candidate_source_from_fn(candidate_fn)
    cfg = config if isinstance(config, EnergyForecastingConfig) else EnergyForecastingConfig.from_mapping(config)
    _trace(cfg, "candidate scoring start")
    t0 = time.time()
    try:
        result = _train_lgbm_score(
            inputs.reset_index(drop=True), pd.Series(outputs).reset_index(drop=True), candidate_fn, cfg
        )
        result.candidate_source = source
        result.runtime_seconds = time.time() - t0
        _trace(
            cfg,
            "candidate scoring done "
            f"status={result.candidate_status} "
            f"score={result.score_for_llmfe:.6f} "
            f"inner_eval_rmse={result.inner_eval_rmse_log1p:.6f} "
            f"best_iter={result.best_iteration} "
            f"runtime={result.runtime_seconds:.1f}s "
            f"n_cols={result.n_candidate_columns}",
        )
    except Exception as exc:
        result = _safe_worst_score(str(exc), candidate_source=source)
        result.runtime_seconds = time.time() - t0
        result.prompt_inputs, result.prompt_outputs = _make_prompt_sample(inputs.reset_index(drop=True), outputs, cfg)
        _trace(
            cfg,
            "candidate scoring failed "
            f"runtime={result.runtime_seconds:.1f}s "
            f"reason={result.details.get('reason')}",
        )
    _write_audit(cfg, result)
    return result


def score_candidate_from_llmfe(
    data: Mapping[str, Any], candidate_fn: CandidateFn
) -> Tuple[float, pd.DataFrame, pd.Series]:
    """LLM-FE ``@evaluate.run`` entry point.

    LLM-FE expects ``(score, input_data, output_data)``.  The score is
    higher-is-better, so the adapter returns negative inner-eval RMSLE/RMSE on
    log1p targets.
    """
    if "inputs" not in data or "outputs" not in data:
        raise KeyError("LLM-FE data dict must contain 'inputs' and 'outputs'.")
    cfg = EnergyForecastingConfig.from_mapping(data.get("config"))
    candidate_source = data.get("_llmfe_candidate_function_source") or data.get("_llmfe_candidate_program_source")
    result = score_candidate_feature(
        data["inputs"], data["outputs"], candidate_fn, cfg, candidate_source=candidate_source
    )
    # return result.score_for_llmfe, result.prompt_inputs, result.prompt_outputs
    return result.score_for_llmfe, data["inputs"], data["outputs"]


def candidate_fn_from_body(body: str) -> CandidateFn:
    """Build a candidate function from a generated body string.

    This helper is intended for local mock testing only.  LLM-FE itself compiles
    generated candidates through its own evaluator/sandbox.
    """
    source = "def modify_features(df_input):\n"
    for line in body.splitlines():
        source += f"    {line}\n" if line.strip() else "\n"
    namespace: Dict[str, Any] = {"pd": pd, "np": np, "math": math}
    exec(source, namespace)
    return namespace["modify_features"]
