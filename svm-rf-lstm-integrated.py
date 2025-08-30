#!/usr/bin/env python3
"""
svm-rf.py - Reproducible script for univariate time-series forecasting
using SVR or RandomForest with time-series cross-validation and parallel grid search.

Features:
- Automatic CPU detection and safe parallel configuration
- Deterministic seeding (when run with PYTHONHASHSEED set in the shell)
- Threadpool control for BLAS libraries
- Clean preprocessing, scaling, and supervised framing (timesteps -> X/y)
- GridSearchCV with loky backend and threadpool_limits for reproducible parallelism
- Artifact saving (models, scaler, predictions, metrics, environment info)
- CLI options for model selection and paths

Usage (example):
    PYTHONHASHSEED=12345 python svm-rf.py --data inflatie-bnr-ro.xlsx --model svr --outdir results

Author: Bogdan Oancea
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Any
import ast
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics as skm
import gc

from joblib import parallel_backend, delayed, Parallel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
from matplotlib.ticker import StrMethodFormatter

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.model_selection import ParameterGrid

# Optional import for LSTM -- lazy import in functions
try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

PARAM_GRIDS = {
    'svr': {
        'regressor__est__C': [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'regressor__est__gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'regressor__est__coef0': [0.0, 0.01, 0.5, 1.0, 2.0, 2.5],
        'regressor__est__epsilon': [0.0, 0.01, 0.05, 0.1, 0.2, 0.3],
        'regressor__est__kernel': ['rbf', 'poly'],
        'regressor__est__degree': [1, 2, 3, 4, 5]
    },
    'rf': {
        'regressor__est__n_estimators': [50, 75, 100, 150, 200],
        'regressor__est__max_depth': [None, 2, 5, 10, 20, 30, 40, 50],
        'regressor__est__min_samples_split': [2, 5, 10, 12, 15, 20],
        'regressor__est__max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    }
}

countries = {
    'BG_CPI': 'Bulgaria',
    'RO_CPI': 'Romania',
    'SK_CPI': 'Slovakia',
    'SL_CPI': 'Slovenia',
    'LT_CPI': 'Lithuania',
    'LV_CPI': 'Latvia',
    'ES_CPI': 'Estonia',
    'PL_CPI': 'Poland',
    'HU_CPI': 'Hungary',
    'CZ_CPI': 'Czechia'
}

GRID_SAMPLE_SIZE = 2000
def set_seeds(seed: int = 12345):
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass


# ---- Utility: parallel configuration function (copied/adapted) ----
def configure_parallelism(max_workers_cap: int = 24,
                          prefer_physical: bool = True,
                          seed: int = 12345,
                          verbose: bool = True) -> Tuple[int, int]:
    """
    Detect CPU resources and configure environment variables for safe multithreading.

    Returns:
        (n_jobs, threads_per_worker)
    """
    try:
        import psutil
    except Exception:
        psutil = None

    # Seed python/NumPy RNGs early
    random.seed(seed)
    np.random.seed(seed)

    logical = os.cpu_count() or 1
    physical = None
    if psutil is not None:
        try:
            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None

    total_cores = physical if (prefer_physical and physical) else logical
    if total_cores is None or total_cores < 1:
        total_cores = 1

    max_workers_cap = max(1, int(max_workers_cap))
    desired_workers = min(max_workers_cap, total_cores)
    threads_per_worker = max(1, total_cores // desired_workers)
    n_jobs = desired_workers
    while n_jobs * threads_per_worker > total_cores:
        if threads_per_worker > 1:
            threads_per_worker -= 1
        else:
            n_jobs -= 1
            if n_jobs < 1:
                n_jobs = 1
                break
    if n_jobs * threads_per_worker > total_cores:
        threads_per_worker = 1
        n_jobs = min(total_cores, max(1, n_jobs))

    # Export env vars so worker processes inherit thread limits
    os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(threads_per_worker)

    if verbose:
        logging.info(f"Detected {logical} logical cores"
                     + (f", {physical} physical cores" if physical else "")
                     + f". Using {n_jobs} workers x {threads_per_worker} threads (<= {total_cores}).")
        logging.info(f"Set OMP/OPENBLAS/MKL/VECLIB threads to {threads_per_worker}.")
        logging.info("Note: For full hash determinism set PYTHONHASHSEED in the shell before starting Python.")

    return int(n_jobs), int(threads_per_worker)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SVR, RF or LSTM on HICP series (quarterly/monthly).")
    parser.add_argument("--data", required=True, type=Path, help="Path to Excel input file")
    parser.add_argument("--sheet", type=str, default=0, help="Sheet name or index for Excel read")
    parser.add_argument("--model", choices=("svr", "rf", "lstm"), default="svr",
                        help="Model to run: 'svr', 'rf', or 'lstm'")
    parser.add_argument("--timesteps", type=int, default=7, help="Number of timesteps (window length)")
    parser.add_argument("--outdir", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--train-start", required=True, type=str, default="2006:Q1", help="Train start (e.g. 2006:Q1)")
    parser.add_argument("--train-end", required=True, type=str, default="2022:Q4", help="Train end (e.g. 2022:Q4)")
    parser.add_argument("--test-start", required=True, type=str, default="2021:Q3", help="Test start (e.g. 2021:Q3)")
    parser.add_argument("--max-workers-cap", type=int, default=24, help="Cap for parallel workers")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed")
    parser.add_argument("--epochs", type=int, default=1000, help="LSTM training epochs")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--multicountry", action="store_true", help="Run multicountry experiment")
    parser.add_argument("--experiment_name", required=True, type=str, help="Experiment name")
    parser.add_argument("--features", required=True, type=str, help="List of features")
    return parser.parse_args(argv)



def setup_logging(outdir: Path, verbose: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logfile = outdir / "run.log"
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(logfile, mode="w", encoding="utf-8")])


def read_and_prepare(df_path: Path) -> pd.DataFrame:
    """Read Excel and prepare the time series DataFrame."""
    df = pd.read_excel(df_path, sheet_name=0)
    # Select columns by positions as in original script (robustify by name fallback)
    try:
        ts = df.iloc[:, [0, 1, 3, 4]].copy()
    except Exception:
        ts = df.copy()
    # Normalize column names
    ts.columns = [str(c).strip() for c in ts.columns]
    # Attempt to rename common Romanian column names to English ones
    rename_map = {
        'quarter': 'Quarter', 'Quarter': 'Quarter',
        'HICP': 'HICP', 'indice sentiment': 'Sentiment', 'indice_sentiment': 'Sentiment',
        'rata somajului': 'Unemployment', 'Unemployment': 'Unemployment'
    }
    ts = ts.rename(columns=lambda c: rename_map.get(c, c))
    # Reorder columns if present
    desired = ["Quarter", "Sentiment", "Unemployment", "HICP"]
    present = [c for c in desired if c in ts.columns]
    ts = ts.loc[:, present].copy()
    ts.set_index('Quarter', inplace=True)
    return ts


def read_and_prepare_multicountry_data(df_path: Path) -> pd.DataFrame:
    """Read Excel and prepare the time series DataFrame."""
    df = pd.read_excel(df_path)
    df.rename(columns={'month': 'Month',
    #                     'esi_bulgaria': 'BG_ESI',
    #                     'esi_romania': 'RO_ESI',
    #                     'esi_slovakia': 'SK_ESI',
    #                     'esi_slovenia': 'SL_ESI',
    #                     'esi_lithuania': 'LT_ESI',
    #                     'esi_latvia': 'LV_ESI',
    #                     'esi_estonia': 'ES_ESI',
    #                     'esi_poland': 'PL_ESI',
    #                     'esi_hungary': 'HU_ESI',
    #                     'esi_czechia': 'CZ_ESI',
    #                     'cpi_bulgaria': 'BG_CPI',
    #                     'cpi_romania': 'RO_CPI',
    #                     'cpi_slovakia': 'SK_CPI',
    #                     'cpi_slovenia': 'SL_CPI',
    #                     'cpi_lithuania': 'LT_CPI',
    #                     'cpi_latvia': 'LV_CPI',
    #                     'cpi_estonia': 'ES_CPI',
    #                     'cpi_poland': 'PL_CPI',
    #                     'cpi_hungary': 'HU_CPI',
    #                     'cpi_czechia': 'CZ_CPI',
    #                     'unempl_bulgaria': 'BG_UNEMPL',
    #                     'unempl_romania': 'RO_UNEMPL',
    #                     'unempl_slovakia': 'SK_UNEMPL',
    #                     'unempl_slovenia': 'SL_UNEMPL',
    #                     'unempl_lithuania': 'LT_UNEMPL',
    #                     'unempl_latvia': 'LV_UNEMPL',
    #                     'unempl_estonia': 'ES_UNEMPL',
    #                     'unempl_poland': 'PL_UNEMPL',
    #                     'unempl_hungary': 'HU_UNEMPL',
    #                     'unempl_czechia': 'CZ_UNEMPL',
                        }, inplace=True)

    ts = df.copy()
    ts.set_index('Month', inplace=True)
    return ts



def create_supervised(arr: np.ndarray, timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create supervised dataset: for window size=timesteps, X contains timesteps-1 values, y the last value.
    arr: shape (n_samples, 1)
    """
    n = arr.shape[0]
    if n < timesteps:
        raise ValueError("Not enough data for the requested number of timesteps.")
    X = []
    y = []
    for i in range(0, n - timesteps + 1):
        window = arr[i:i + timesteps, 0]
        X.append(window[:-1])
        y.append(window[-1])
    return np.array(X), np.array(y).reshape(-1, 1)


def create_multivariate_windows(data: np.ndarray, timesteps: int, target_col: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised windows from multivariate array.
    data: ndarray shape (n_rows, n_features)
    timesteps: number of rows in a window (includes the target row)
    target_col: index of the target column inside `data` (0-based)
    Returns:
      X: shape (n_samples, (timesteps-1) * n_features)  -- flattened past values
      y: shape (n_samples, 1)  -- target values (raw)
    """
    n_rows, n_features = data.shape
    if n_rows < timesteps:
        return np.empty((0, (timesteps - 1) * n_features)), np.empty((0, 1))
    n_samples = n_rows - timesteps + 1
    X = np.empty((n_samples, (timesteps - 1) * n_features), dtype=float)
    y = np.empty((n_samples, 1), dtype=float)
    for i in range(n_samples):
        window = data[i: i + timesteps, :]  # shape (timesteps, n_features)
        past = window[: timesteps - 1, :]  # shape (timesteps-1, n_features)
        X[i, :] = past.reshape(-1)  # time-major flatten: t0f0,t0f1,...tNfM
        y[i, 0] = window[timesteps - 1, target_col]  # target is last row's target_col
    return X, y


# LSTM specific 3D window builder (works for univariate and multivariate)

def create_multivariate_3d_windows(data: np.ndarray, timesteps: int, target_col: int = 0) -> Tuple[
    np.ndarray, np.ndarray]:
    n_rows, n_features = data.shape
    if n_rows < timesteps:
        return np.empty((0, timesteps - 1, n_features)), np.empty((0, 1))
    n_samples = n_rows - timesteps + 1
    X = np.zeros((n_samples, timesteps - 1, n_features), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    for i in range(n_samples):
        window = data[i:i + timesteps, :]
        X[i] = window[:timesteps - 1, :]
        y[i, 0] = window[timesteps - 1, target_col]
    return X, y


def save_json_excel(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    # If obj is a dict or list, convert to DataFrame
    if isinstance(obj, dict):
        df = pd.DataFrame([obj])  # single row
    elif isinstance(obj, list):
        df = pd.DataFrame(obj)  # list of dicts
    else:
        raise ValueError("Object must be dict or list of dicts to save as Excel")

    df.to_excel(path.with_suffix(".xlsx"), index=False)


# Minimal filename helpers to ensure consistent artifact naming
def artifact_prefix(outdir: Path, model: str, experiment_name: str) -> Path:
    return outdir / f"{model}_{experiment_name}"


def save_predictions(prefix: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    np.savetxt(prefix.with_name(prefix.name + "_y_true.csv"), y_true, delimiter=",")
    np.savetxt(prefix.with_name(prefix.name + "_y_pred.csv"), y_pred, delimiter=",")


# --------------------------- Experiment runner (central) ---------------------------

def build_lstm(input_shape: Tuple[int, int],
               units: int = 32,
               dropout1: float = 0.0,
               recurrent_dropout1:float=0.0,
               dropout2: float = 0.0,
               recurrent_dropout2:float=0.0,
               lr: float = 1e-3,
               activation: str = 'relu',
               l2_reg: float = 1e-5) -> Any:
    """
    Build a simple stacked LSTM -> Dense(1) regressor.
    l2_reg: L2 regularization strength applied to the final dense layer's kernel.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available; install tensorflow to run LSTM experiments.")
    inputs = keras.Input(shape=input_shape, name="X")
    x = inputs
    x = keras.layers.LSTM(units,
                          return_sequences=True,
                          dropout=dropout1,
                          recurrent_dropout = recurrent_dropout1,
                          activation=activation)(x)
    x = keras.layers.LSTM(units,
                          dropout=dropout2,
                          recurrent_dropout = recurrent_dropout2,
                          activation=activation)(x)
    outputs = keras.layers.Dense(
        1,
        activation='linear',
        name='out',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse', metrics=[keras.metrics.MeanAbsoluteError()])
    return model

def run_multi_country_experiment(
        experiment_name: str,
        features: str,
        target_col_idx: int,
        ts: pd.DataFrame,
        train_start: str,
        train_end: str,
        test_start: str,
        args,
        out_dir: Path,
        n_jobs: int,
        threads_per_worker: int,
        tscv: TimeSeriesSplit,
):
    for col, country in countries.items():
        logging.info("Starting multicountry experiment '%s' with features=%s (target_idx=%d) using model=%s, for country %s",
                     experiment_name, features, target_col_idx, args.model, country)
        new_name = f"{experiment_name}_{country}"
        new_features=[f + f'_{country}' for f in ast.literal_eval(features)]
        new_features = list(map(str.lower, new_features))
        print("new features are:")
        print(new_features)
        run_experiment(
            experiment_name=new_name,
            features=new_features,
            target_col_idx=0,
            ts=ts,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            args=args,
            out_dir=out_dir,
            n_jobs=n_jobs,
            threads_per_worker=threads_per_worker,
            tscv=tscv
        )
def run_experiment(
        experiment_name: str,
        features: str,
        target_col_idx: int,
        ts: pd.DataFrame,
        train_start: str,
        train_end: str,
        test_start: str,
        args,
        out_dir: Path,
        n_jobs: int,
        threads_per_worker: int,
        tscv: TimeSeriesSplit,
):
    """Unified experiment runner that supports 'svr', 'rf' and 'lstm'.

    Artifacts are saved under out_dir / f"{args.model}_{experiment_name}".
    Returns a metrics dict (or None on failure).
    """
    logging.info("Starting experiment '%s' with features=%s (target_idx=%d) using model=%s",
                 experiment_name, features, target_col_idx, args.model)



    # Prepare train/test DataFrames
    if not args.multicountry:
        new_features = [f for f in ast.literal_eval(features)]
        new_features = list(new_features)
        features = new_features
    print("new features are:")
    print(features)
    train_df = ts.loc[(ts.index >= train_start) & (ts.index <= train_end), features].copy().dropna()
    test_df = ts.loc[(ts.index >= test_start), features].copy().dropna()
    logging.info("[%s] Raw shapes: train=%s test=%s", experiment_name, train_df.shape, test_df.shape)

    # Window creation depends on whether we're running LSTM (3D) or classical ML (2D)
    is_lstm = (args.model == 'lstm')

    if is_lstm:
        X_train_raw, y_train_raw = create_multivariate_3d_windows(train_df.values.astype(np.float32), args.timesteps,
                                                                  target_col=target_col_idx)
        X_test_raw, y_test_raw = create_multivariate_3d_windows(test_df.values.astype(np.float32), args.timesteps,
                                                                target_col=target_col_idx)
    else:
        # classical 2D feature matrices
        if len(features) == 1:
            X_train_raw, y_train_raw = create_supervised(train_df[[features[0]]].values.reshape(-1, 1), args.timesteps)
            X_test_raw, y_test_raw = create_supervised(test_df[[features[0]]].values.reshape(-1, 1), args.timesteps)
        else:
            X_train_raw, y_train_raw = create_multivariate_windows(train_df.values, args.timesteps,
                                                                   target_col=target_col_idx)
            X_test_raw, y_test_raw = create_multivariate_windows(test_df.values, args.timesteps,
                                                                 target_col=target_col_idx)

    logging.info("[%s] Window shapes: X_train=%s y_train=%s X_test=%s y_test=%s",
                 experiment_name, getattr(X_train_raw, 'shape', None), getattr(y_train_raw, 'shape', None),
                 getattr(X_test_raw, 'shape', None), getattr(y_test_raw, 'shape', None))

    if (not is_lstm and X_train_raw.shape[0] == 0) or (is_lstm and X_train_raw.size == 0):
        logging.error("[%s] Not enough data for windows; aborting this experiment.", experiment_name)
        return None

    # Save input hashes for reproducibility
    def _hash(a: np.ndarray) -> str:
        return hashlib.md5(a.tobytes()).hexdigest()

    save_json_excel({
        f"{experiment_name}_x_train_hash": _hash(X_train_raw),
        f"{experiment_name}_y_train_hash": _hash(y_train_raw),
        f"{experiment_name}_x_test_hash": _hash(X_test_raw),
        f"{experiment_name}_y_test_hash": _hash(y_test_raw),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__
    }, out_dir / f"input_hashes_{experiment_name}.json")

    prefix = out_dir / f"{args.model}_{experiment_name}"
    prefix.mkdir(parents=True, exist_ok=True)

    # --- Branch for classical ML models (SVR / RF) ---
    if not is_lstm:
        grid_key = f"{args.model}"
        param_grid = PARAM_GRIDS.get(grid_key, {})

        # Build pipeline/regressor
        if len(features) == 1:
            if args.model == "svr":
                base_pipeline = Pipeline([('scaler', MinMaxScaler()), ('est', SVR())])
            else:
                rf = RandomForestRegressor(n_jobs=1, random_state=args.seed)
                base_pipeline = Pipeline([('scaler', MinMaxScaler()), ('est', rf)])
            regressor = TransformedTargetRegressor(regressor=base_pipeline, transformer=MinMaxScaler())
        else:
            n_features = len(features)
            T_minus1 = args.timesteps - 1

            def _get_feature_col_groups(n_features: int, timesteps_minus_one: int):
                groups = []
                for feat_j in range(n_features):
                    cols = [feat_j + i * n_features for i in range(timesteps_minus_one)]
                    groups.append(cols)
                return groups

            feature_col_groups = _get_feature_col_groups(n_features=n_features, timesteps_minus_one=T_minus1)
            ct_transformers = [(f"feat{i}", StandardScaler(), cols) for i, cols in enumerate(feature_col_groups)]
            col_transformer = ColumnTransformer(ct_transformers, remainder='drop', sparse_threshold=0)

            if args.model == "svr":
                base_pipeline = Pipeline([('pre', col_transformer), ('est', SVR())])
            else:
                rf = RandomForestRegressor(n_jobs=1, random_state=args.seed)
                base_pipeline = Pipeline([('pre', col_transformer), ('est', rf)])
            regressor = TransformedTargetRegressor(regressor=base_pipeline, transformer=MinMaxScaler())

        logging.info("[%s] Starting GridSearch with n_jobs=%d threads_per_worker=%d", experiment_name, n_jobs,
                     threads_per_worker)
        grid = None
        with parallel_backend('loky', n_jobs=n_jobs):
            with threadpool_limits(limits=threads_per_worker):
                try:
                    grid = GridSearchCV(regressor, param_grid, cv=tscv,
                                        scoring='neg_mean_squared_error', n_jobs=n_jobs,
                                        pre_dispatch='n_jobs', refit=True, verbose=1)
                    grid.fit(X_train_raw, y_train_raw)
                except Exception as e:
                    logging.exception("[%s] GridSearchCV failed: %s", experiment_name, e)
                    raise

        logging.info("[%s] GridSearch best params: %s", experiment_name, grid.best_params_)
        save_json_excel(grid.best_params_, prefix / f"{args.model}_best_params.json")
        joblib.dump(grid.best_estimator_, prefix / f"{args.model}_{experiment_name}.pkl")

        # Predictions & metrics
        y_train_pred = grid.predict(X_train_raw).reshape(-1, 1)
        y_test_pred = grid.predict(X_test_raw).reshape(-1, 1)
        y_train_true = y_train_raw
        y_test_true = y_test_raw

        save_predictions(prefix / f"{args.model}_{experiment_name}", y_train_true, y_train_pred)
        save_predictions(prefix / f"{args.model}_{experiment_name}_test", y_test_true, y_test_pred)

        metrics = {
            "MAPE_train": float(skm.mean_absolute_percentage_error(y_train_true, y_train_pred) * 100),
            "MSE_train": float(skm.mean_squared_error(y_train_true, y_train_pred)),
            "MAE_train": float(skm.mean_absolute_error(y_train_true, y_train_pred)),
            "MAPE_test": float(skm.mean_absolute_percentage_error(y_test_true, y_test_pred) * 100),
            "MSE_test": float(skm.mean_squared_error(y_test_true, y_test_pred)),
            "MAE_test": float(skm.mean_absolute_error(y_test_true, y_test_pred))
        }
        save_json_excel(metrics, prefix / f"{args.model}_{experiment_name}_metrics.json")
        logging.info("[%s] Metrics: %s", experiment_name, metrics)

        plot_name = f"{args.model}_{experiment_name}"
        plot_predictions(ts, train_start, train_end, y_train_true, y_train_pred, y_test_true, y_test_pred,
                         out_dir, plot_name, step=4, fontsize=12, timesteps=args.timesteps)


    # --- Branch for LSTM ---
    else:
        if not TF_AVAILABLE:
            logging.error("[%s] TensorFlow not available; cannot run LSTM experiment.", experiment_name)
            return None

        # 1) determinism and threads (avoid CPU oversubscription)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        # 2) enable mixed precision if GPU is present
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy('mixed_float16')
                logging.info("Using mixed_float16 policy for speed.")
        except Exception:
            pass

        logging.info("[%s] Running LSTM cross-validated grid search (parallel over param combos)", experiment_name)
        set_seeds(args.seed)

        # grid (leaner, sensible defaults)
        grid_space = {
            'units': [128, 256, 512],  # smaller but efficient
            'dropout1': [0.0, 0.1, 0.2],  # keep only input dropout
            'recurrent_dropout1': [0.0, 0.1, 0.2],
            'dropout2': [0.0, 0.1, 0.2],  # usually redundant for 1-layer LSTM
            'recurrent_dropout2': [0.0, 0.1, 0.2],
            'l2_reg': [0.0, 1e-4, 1e-3],
            'lr': [3e-4, 1e-3, 3e-3],  # log-ish grid; avoid 1e-1
            'batch_size': [1, 8, 16],  # larger batches
        }
        param_list_full = list(ParameterGrid(grid_space))
        # Randomly sample K candidates instead of full cartesian product
        max_candidates = min(GRID_SAMPLE_SIZE, len(param_list_full))  # tune as you like
        rng = np.random.default_rng(seed=args.seed or 0)
        param_list_idx = rng.choice(len(param_list_full), size=max_candidates, replace=False)
        param_list = [param_list_full[i] for i in param_list_idx]
        logging.info("[%s] Parameter grid sampled: %d/%d combinations",
                     experiment_name, len(param_list), len(param_list_full))

        # basic shapes
        X_all = X_train_raw  # shape (n_windows, timesteps-1, n_features)
        y_all = y_train_raw  # shape (n_windows, 1)
        X_test = X_test_raw
        y_test = y_test_raw

        n_features = X_all.shape[2]
        tscv_local = TimeSeriesSplit(n_splits=3)

        def _fit_scalers_vectorized(X_tr_3d, y_tr_2d):
            """
            X_tr_3d: (n_windows, T, F); y_tr_2d: (n_windows, 1)
            Returns (x_mean, x_std, y_min, y_max) to use for train/val transform.
            """
            # flatten time dimension; keep features separate
            Xf = X_tr_3d.reshape(-1, n_features).astype(np.float32, copy=False)
            x_mean = Xf.mean(axis=0)
            x_std = Xf.std(axis=0)
            x_std[x_std == 0] = 1.0

            y_min = float(np.min(y_tr_2d))
            y_max = float(np.max(y_tr_2d))
            if y_max == y_min:
                y_max = y_min + 1.0
            return x_mean, x_std, y_min, y_max

        def _transform_X(X_3d, x_mean, x_std):
            # broadcast (n, T, F) with (F,)
            return ((X_3d.astype(np.float32) - x_mean) / x_std).astype(np.float32, copy=False)

        def _transform_y_minmax(y_2d, y_min, y_max):
            return ((y_2d.astype(np.float32) - y_min) / (y_max - y_min)).astype(np.float32, copy=False)

        def _make_dataset(X, y, batch_size, training=True):
            ds = tf.data.Dataset.from_tensor_slices((X, y))
            #if training:
            #   ds = ds.shuffle(min(len(X), 4096), seed=args.seed or 0, reshuffle_each_iteration=True)
            ds = ds.batch(batch_size, drop_remainder=False)
            ds = ds.cache().prefetch(tf.data.AUTOTUNE)
            return ds


        # helper: fit scalers inside each fold and train model, returning validation loss for that fold
        def eval_fold(params, train_idx, val_idx, seed_base):
            # Seeds per fold (your approach preserved)
            set_seeds(
                seed_base + (int(hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest(), 16) % 10_000))

            X_tr_fold, y_tr_fold = X_all[train_idx], y_all[train_idx]
            X_val_fold, y_val_fold = X_all[val_idx], y_all[val_idx]

            # Vectorized scaling fit on TRAIN ONLY
            x_mean, x_std, y_min, y_max = _fit_scalers_vectorized(X_tr_fold, y_tr_fold)
            X_tr_s = _transform_X(X_tr_fold, x_mean, x_std)
            X_val_s = _transform_X(X_val_fold, x_mean, x_std)
            y_tr_s = _transform_y_minmax(y_tr_fold, y_min, y_max)
            y_val_s = _transform_y_minmax(y_val_fold, y_min, y_max)

            # tf.data pipelines
            bs = int(params['batch_size'])
            ds_tr = _make_dataset(X_tr_s, y_tr_s, bs, training=True)
            ds_val = _make_dataset(X_val_s, y_val_s, bs, training=False)

            try:
                model = build_lstm(
                    input_shape=(args.timesteps - 1, n_features),
                    units=int(params['units']),
                    dropout1=float(params['dropout1']),
                    recurrent_dropout1=(params['recurrent_dropout1']),
                    dropout2=float(params['dropout2']),
                    recurrent_dropout2=(params['recurrent_dropout2']),
                    l2_reg=float(params.get('l2_reg', 0.0) or 0.0),
                    lr=float(params['lr'])
                )
            except Exception as e:
                logging.exception("[%s] build_lstm failed for params %s: %s", experiment_name, params, e)
                return np.inf

            # Shorter patience; stop early across CV
            cb = [
                keras.callbacks.EarlyStopping(monitor='loss', patience=8, start_from_epoch = 50, restore_best_weights=True, verbose=0),
                keras.callbacks.TerminateOnNaN(),
            ]

            try:
                hist = model.fit(
                    ds_tr,
                    validation_data=ds_val,
                    epochs=min(getattr(args, 'epochs', 700), 500),  # tighter cap during CV
                    verbose=0 if not getattr(args, 'verbose', False) else 2,
                    callbacks=cb
                )
                val_loss = float(np.min(hist.history.get('val_loss', [np.inf])))
            except Exception:
                logging.exception("[%s] Training failed in fold for params %s", experiment_name, params)
                val_loss = np.inf
            finally:
                try:
                    keras.backend.clear_session()
                except Exception:
                    pass
                del model, hist, X_tr_fold, X_val_fold, X_tr_s, X_val_s, y_tr_s, y_val_s
                gc.collect()

            return val_loss

        # evaluate one params set: run CV sequentially (inside same process)
        def evaluate_params(params):
            seed_base = args.seed or 0
            fold_losses = []
            for fold_i, (train_idx, val_idx) in enumerate(tscv_local.split(X_all)):
                fold_losses.append(eval_fold(params, train_idx, val_idx, seed_base + fold_i * 7))
            mean_loss = float(np.mean(fold_losses)) if fold_losses else np.inf
            logging.info("[%s] Params %s -> mean_val_loss=%.6f (folds=%d)", experiment_name, params, mean_loss,
                         len(fold_losses))
            return {'params': params, 'mean_val_loss': mean_loss}


        #results_list = [evaluate_params(p) for p in param_list]
        with threadpool_limits(limits=1):
            results_list = Parallel(
                n_jobs=n_jobs,  # all logical cores
                backend="loky",
                verbose=10
            )(delayed(evaluate_params)(p) for p in param_list)

        # select best, then retrain on full data (vectorized scaling & tf.data)
        valid_results = [r for r in results_list if np.isfinite(r['mean_val_loss'])]
        if not valid_results:
            logging.error("[%s] No valid parameter configuration succeeded during CV.", experiment_name)
            return None

        best_res = min(valid_results, key=lambda r: r['mean_val_loss'])
        best_params = best_res['params']
        logging.info("[%s] Best params after CV: %s (mean_val_loss=%.6f)", experiment_name, best_params,
                     best_res['mean_val_loss'])

        # --- final fit on all training windows ---
        # Vectorized full-fit scalers
        Xf = X_all.reshape(-1, n_features).astype(np.float32)
        x_mean_all = Xf.mean(axis=0);
        x_std_all = Xf.std(axis=0);
        x_std_all[x_std_all == 0] = 1.0

        y_min_all = float(np.min(y_all));
        y_max_all = float(np.max(y_all))
        if y_max_all == y_min_all: y_max_all = y_min_all + 1.0

        X_all_s = _transform_X(X_all, x_mean_all, x_std_all)
        y_all_s = _transform_y_minmax(y_all, y_min_all, y_max_all)

        if X_test.shape[0] > 0:
            X_test_s = _transform_X(X_test, x_mean_all, x_std_all)
        else:
            X_test_s = np.empty((0, args.timesteps - 1, n_features), dtype=np.float32)

        final_model = build_lstm(
            input_shape=(args.timesteps - 1, n_features),
            units=int(best_params['units']),
            dropout1=float(best_params['dropout1']),
            recurrent_dropout1=float(best_params['recurrent_dropout1']),
            dropout2=float(best_params['dropout2']),
            recurrent_dropout2=float(best_params['recurrent_dropout2']),
            l2_reg=float(best_params.get('l2_reg', 0.0) or 0.0),
            lr=float(best_params['lr'])
        )

        # callbacks_final = [
        #     keras.callbacks.EarlyStopping(monitor='loss', patience=8, restore_best_weights=True, verbose=0),
        #     keras.callbacks.TerminateOnNaN(),
        # ]
        early_final = keras.callbacks.EarlyStopping(
            monitor="loss", patience=25, start_from_epoch = 250, restore_best_weights=True, verbose = 0
        )
        bs = int(best_params['batch_size'])
        ds_full = _make_dataset(X_all_s, y_all_s, bs, training=True)
        save_json_excel(best_params, prefix / "lstm_best_params.json")
        hist = final_model.fit(ds_full, epochs=1000,
                        verbose=1 if getattr(args, 'verbose', False) else 0,
                        callbacks=early_final)

        epochs_run = len(hist.epoch)  # real number of epochs executed in this fit
        monitor = getattr(early_final, "monitor", "loss")
        if monitor not in hist.history:
            monitor = "loss"
        best_epoch = int(np.argmin(hist.history[monitor])) + 1
        best_value = float(np.min(hist.history[monitor]))
        stopped_epoch = getattr(early_final, "stopped_epoch", 0)  # >0 if early stopping triggered
        summary = {
            "epochs_run": int(epochs_run),
            "best_epoch": int(best_epoch),
            "monitor": str(monitor),
            "best_value": float(best_value),
            "early_stopped": bool(stopped_epoch > 0),
            "stopped_epoch": int(stopped_epoch),
        }

        save_json_excel(summary, prefix / "lstm_epochs.json" )
        print(f"Ran {epochs_run} epochs; best @ epoch {best_epoch} ({monitor}={best_value:.6f}); "
              f"early_stopped={'yes' if stopped_epoch > 0 else 'no'}")

        # predictions (ensure float32 outputs under mixed precision)
        y_train_pred_s = final_model.predict(_make_dataset(X_all_s, y_all_s, bs, training=False), verbose=0).astype(
            np.float32)
        y_train_true = y_all.astype(np.float32)
        y_train_pred = (y_train_pred_s * (y_max_all - y_min_all)) + y_min_all

        if X_test_s.shape[0] > 0:
            y_test_pred_s = final_model.predict(
                tf.data.Dataset.from_tensor_slices(X_test_s).batch(bs).prefetch(tf.data.AUTOTUNE), verbose=0).astype(
                np.float32)
            y_test_pred = (y_test_pred_s * (y_max_all - y_min_all)) + y_min_all
            y_test_true = y_test.astype(np.float32)
        else:
            y_test_pred = np.empty((0, 1), dtype=np.float32)
            y_test_true = np.empty((0, 1), dtype=np.float32)

        # Save final artifacts
        final_model.save(prefix / "lstm_final_model.keras")
        save_predictions(prefix / "lstm_train", y_train_true, y_train_pred)
        if y_test_pred.shape[0] > 0:
            save_predictions(prefix / "lstm_test", y_test_true, y_test_pred)

        # metrics
        metrics = {
            'train_MAPE': float(skm.mean_absolute_percentage_error(y_train_true, y_train_pred) * 100),
            'train_MSE': float(skm.mean_squared_error(y_train_true, y_train_pred)),
            'train_MAE': float(skm.mean_absolute_error(y_train_true, y_train_pred))
        }
        if y_test_pred.shape[0] > 0:
            metrics.update({
                'test_MAPE': float(skm.mean_absolute_percentage_error(y_test_true, y_test_pred) * 100),
                'test_MSE': float(skm.mean_squared_error(y_test_true, y_test_pred)),
                'test_MAE': float(skm.mean_absolute_error(y_test_true, y_test_pred))
            })
        else:
            metrics.update({'test_MAPE': None, 'test_MSE': None, 'test_MAE': None})


        save_json_excel(metrics, prefix / "lstm_metrics.json")
        logging.info("[%s] LSTM CV experiment done. Best params: %s. Metrics: %s", experiment_name, best_params, metrics)
        return metrics


def plot_hicp_series(ts: pd.DataFrame, out_dir: Path, fontsize: int = 12, step: int = 4) -> None:
    """
    Plot full HICP series; put xticks every `step` items from the combined index,
    keep matplotlib's label formatting, avoid unlabeled minor ticks.
    """
    ax = ts.plot(y='HICP', subplots=True, figsize=(15, 8), fontsize=fontsize)
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.08)
    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(right=0.95)
    plt.xlabel('Quarter', fontsize=fontsize + 1)
    plt.ylabel('HICP', fontsize=fontsize + 1)
    plt.title('Romanian HICP')
    for subplot in ax:  # Adjust for each subplot if subplots=True
        ticks = list(range(0, len(ts.index), step))  # Set tick positions every step index
        if len(ts.index) - 1 not in ticks:  # Ensure the last index is included
            ticks.append(len(ts.index) - 1)
        labels = ts.index[ticks]  # Use the corresponding indices for labels
        subplot.set_xticks(ticks)  # Apply the tick positions
        subplot.set_xticklabels(labels, rotation=90, ha='center')  # Apply the tick labels
        subplot.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    # Save the figure
    fig_path = out_dir / "HICP.eps"
    try:
        plt.savefig(fig_path, format='eps', dpi=1200)
        logging.info("Saved series plot to %s", fig_path)
    finally:
        for subplot in ax:
            plt.close(subplot.get_figure())


def plot_hicp_train_test_series(ts: pd.DataFrame, train_start: str, train_end: str, out_dir: Path, fontsize: int = 12,
                                step: int = 4) -> None:
    train_timestamps = ts[(ts.index <= train_end) & (ts.index >= train_start)].index[0:]
    test_timestamps = ts[(ts.index > train_end)].index[0:]
    y_tr = ts.iloc[(ts.index <= train_end) & (ts.index >= train_start)][['HICP']]
    y_ts = ts.iloc[ts.index > train_end][['HICP']]

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.rcParams.update({'font.size': fontsize + 1})
    ax.plot(train_timestamps, y_tr, color='red', linewidth=2.0, alpha=0.6, label='Train Actual')
    ax.plot(test_timestamps, y_ts, color='purple', linewidth=2.0, alpha=0.6, label='Test Actual')

    ticks = list(range(0, len(ts.index), step))  # Set tick positions every step^th index
    if len(ts.index) - 1 not in ticks:  # Ensure the last index is included
        ticks.append(len(ts.index) - 1)

    labels = ts.index[ticks]  # Use the corresponding indices for labels
    ax.set_xticks(ticks)  # Apply the tick positions
    ax.set_xticklabels(labels, rotation=90, ha='center')  # Apply the tick labels
    ax.legend()
    ax.set_xlabel('Quarter')
    ax.set_ylabel('HICP')
    ax.set_title("HICP train / test split")
    # Remove minor ticks to avoid stray unlabeled marks
    ax.xaxis.set_minor_locator(plt.NullLocator())

    plt.tight_layout()
    # Save the figure
    fig_path = out_dir / "HICP_train_test.eps"
    try:
        plt.savefig(fig_path, format='eps', dpi=1200)
        logging.info("Saved train  / test series plot to %s", fig_path)
    finally:
        plt.close(ax.get_figure())

    logging.info("Saved prediction plot to %s", fig_path)


def plot_predictions(ts: pd.DataFrame, train_start: str, train_end: str, y_train: np.ndarray, y_train_pred: np.ndarray,
                     y_test: np.ndarray, y_test_pred: np.ndarray, out_dir: Path, model: str, step: int = 4,
                     fontsize: int = 12, timesteps: int = 7) -> None:
    train_timestamps = ts[(ts.index <= train_end) & (ts.index >= train_start)].index[timesteps - 1:]
    test_timestamps = ts[(ts.index > train_end)].index[0:]

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.rcParams.update({'font.size': fontsize + 1})
    ax.plot(train_timestamps, y_train, color='red', linewidth=2.0, alpha=0.6, label='Train Actual')
    ax.plot(train_timestamps, y_train_pred, color='blue', linewidth=1, label='Train Predicted')
    ax.plot(test_timestamps, y_test, color='purple', linewidth=2.0, alpha=0.6, label='Test Actual')
    ax.plot(test_timestamps, y_test_pred, color='navy', linewidth=1, label='Test Predicted')

    ticks = list(range(timesteps - 1, len(ts.index), step))  # Set tick positions every step^th index
    if len(ts.index) - 1 not in ticks:  # Ensure the last index is included
        ticks.append(len(ts.index) - 1)

    labels = ts.index[ticks]  # Use the corresponding indices for labels
    ticks = [t - timesteps + 1 for t in ticks]
    ax.set_xticks(ticks)  # Apply the tick positions
    ax.set_xticklabels(labels, rotation=90, ha='center')  # Apply the tick labels
    ax.legend()
    ax.set_xlabel('Quarter')
    ax.set_ylabel('HICP')
    ax.set_title(f"{model.upper()} predicted vs. actual values")
    # Remove minor ticks to avoid stray unlabeled marks
    ax.xaxis.set_minor_locator(plt.NullLocator())

    plt.tight_layout()
    fig_path = out_dir / f"{model}_pred_vs_actual.eps"
    try:
        plt.savefig(fig_path, format='eps', dpi=1200)
        logging.info("%s: saved predicted vs. actual values plot to", fig_path)
    finally:
        plt.close(fig)

    logging.info("Saved prediction plot to %s", fig_path)


# ---- Main pipeline ----
def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(f"results_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, args.verbose)
    logging.info("Starting run with args: %s", vars(args))

    n_jobs, threads_per_worker = configure_parallelism(max_workers_cap=args.max_workers_cap,
                                                       prefer_physical=True,
                                                       seed=args.seed,
                                                       verbose=args.verbose)

    if args.multicountry :
        ts = read_and_prepare_multicountry_data(args.data)
    else :
        ts = read_and_prepare(args.data)


    # if 'HICP' not in ts.columns:
    #     if not any(col in ts.columns for col, _ in countries.items()):
    #         logging.error("Neither 'HICP' nor any country-specific columns (%s) found. Available: %s",
    #                       list(countries.keys()), ts.columns.tolist())
    #         raise SystemExit(1)

    # quick plots
    if not args.multicountry:
        plot_hicp_series(ts, out_dir, 12, 4)
        plot_hicp_train_test_series(ts, args.train_start, args.train_end, out_dir, 12, 4)


    train_start = args.train_start
    train_end = args.train_end
    test_start = args.test_start

    # CV
    tscv = TimeSeriesSplit(n_splits=3)
    if not args.multicountry:
        run_experiment(
            experiment_name=args.experiment_name,
            features=args.features,
            target_col_idx=0,
            ts=ts,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            args=args,
            out_dir=out_dir,
            n_jobs=n_jobs,
            threads_per_worker=threads_per_worker,
            tscv=tscv
        )
    else :
        run_multi_country_experiment(
            experiment_name=args.experiment_name,
            features=args.features,
            target_col_idx=0,
            ts=ts,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            args=args,
            out_dir=out_dir,
            n_jobs=n_jobs,
            threads_per_worker=threads_per_worker,
            tscv=tscv
        )
    # Run univariate experiment
    # run_experiment(
    #     experiment_name="univariate",
    #     features=['HICP'],
    #     target_col_idx=0,
    #     ts=ts,
    #     train_start=train_start,
    #     train_end=train_end,
    #     test_start=test_start,
    #     args=args,
    #     out_dir=out_dir,
    #     n_jobs=n_jobs,
    #     threads_per_worker=threads_per_worker,
    #     tscv=tscv
    # )
    # keras.backend.clear_session()
    # gc.collect()
    #
    # run_experiment(
    #     experiment_name="multivariate",
    #     features=['HICP', 'Sentiment', 'Unemployment'],
    #     target_col_idx=0,
    #     ts=ts,
    #     train_start=train_start,
    #     train_end=train_end,
    #     test_start=test_start,
    #     args=args,
    #     out_dir=out_dir,
    #     n_jobs=n_jobs,
    #     threads_per_worker=threads_per_worker,
    #     tscv=tscv
    # )
    # keras.backend.clear_session()
    # gc.collect()
    #
    # run_experiment(
    #     experiment_name="multivariateHS",
    #     features=['HICP', 'Sentiment'],
    #     target_col_idx=0,
    #     ts=ts,
    #     train_start=train_start,
    #     train_end=train_end,
    #     test_start=test_start,
    #     args=args,
    #     out_dir=out_dir,
    #     n_jobs=n_jobs,
    #     threads_per_worker=threads_per_worker,
    #     tscv=tscv
    # )
    # keras.backend.clear_session()
    # gc.collect()
    # run_experiment(
    #     experiment_name="multivariateHU",
    #     features=['HICP', 'Unemployment'],
    #     target_col_idx=0,
    #     ts=ts,
    #     train_start=train_start,
    #     train_end=train_end,
    #     test_start=test_start,
    #     args=args,
    #     out_dir=out_dir,
    #     n_jobs=n_jobs,
    #     threads_per_worker=threads_per_worker,
    #     tscv=tscv
    # )

    # run_multi_country_experiment(
    #     experiment_name="univariate",
    #     features=['CPI'],
    #     target_col_idx=0,
    #     ts=ts,
    #     train_start=train_start,
    #     train_end=train_end,
    #     test_start=test_start,
    #     args=args,
    #     out_dir=out_dir,
    #     n_jobs=n_jobs,
    #     threads_per_worker=threads_per_worker,
    #     tscv=tscv
    # )
    logging.info("Script finished.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()


# Define train/test split for multicountry data
# train_start = '2006-01'
# train_end = '2024-12'
# test_start = '2024-07'

# Define train/test split for RO data
#train-start="2006:Q1"
#train-end="2022:Q4"
#test-start="2021:Q3"

# command lines multicountry
# python --data=dateCEE-hicp.xlsx --model=svr --train-start=2006-01 --train-end=2024-12 --test-start=2024-07 --multicountry -features=['CPI'] --experiment_name=univariate
# python --data=dateCEE-hicp.xlsx --model=svr --train-start=2006-01 --train-end=2024-12 --test-start=2024-07 --multicountry -features=['CPI', 'ESI'] --experiment_name=multivaritateCPI_ESI
# python --data=dateCEE-hicp.xlsx --model=svr --train-start=2006-01 --train-end=2024-12 --test-start=2024-07 --multicountry -features=['CPI', 'UNEMPL'] --experiment_name=multivaritateCPI_UNEMPL
# python --data=dateCEE-hicp.xlsx --model=svr --train-start=2006-01 --train-end=2024-12 --test-start=2024-07 --multicountry -features=['CPI', 'ESI', 'UNEMPL'] --experiment_name=multivaritateCPI_ESI_UNEMPL
# python --data=dateCEE-hicp.xlsx --model=rf --train-start=2006-01 --train-end=2024-12 --test-start=2024-07 --multicountry -features=['CPI'] --experiment_name=univariate
# python --data=dateCEE-hicp.xlsx --model=rf --train-start=2006-01 --train-end=2024-12 --test-start=2024-07 --multicountry -features=['CPI', 'ESI'] -experiment_name=multivaritateCPI_ESI
# python --data=dateCEE-hicp.xlsx --model=rf --train-start=2006-01 --train-end=2024-12 --test-start=2024-07 --multicountry -features=['CPI', 'UNEMPL'] -experiment_name=multivaritateCPI_UNEMPL
# python --data=dateCEE-hicp.xlsx --model=rf --train-start=2006-01 --train-end=2024-12 --test-start=2024-07 --multicountry -features=['CPI', 'ESI', 'UNEMPL'] --experiment_name=multivaritateCPI_ESI_UNEMPL
# python --data=dateCEE-hicp.xlsx --model=lstm --train-start=2006-01 --train-end=2024-12 --test-start=2024-07 --multicountry -features=['CPI'] --experiment_name=univariate
# python --data=dateCEE-hicp.xlsx --model=lstm --train-start=2006-01 --train-end=2024-12 --test-start=2024-07 --multicountry -features=['CPI', 'ESI'] --experiment_name=multivaritateCPI_ESI
# python --data=dateCEE-hicp.xlsx --model=lstm --train-start=2006-01 --train-end=2024-12 --test-start=2024-07 --multicountry -features=['CPI', 'UNEMPL'] --experiment_name=multivaritateCPI_UNEMPL
# python --data=dateCEE-hicp.xlsx --model=lstm --train-start=2006-01 --train-end=2024-12 --test-start=2024-07 --multicountry -features=['CPI', 'ESI', 'UNEMPL'] --experiment_name=multivaritateCPI_ESI_UNEMPL

# command lines RO ['HICP', 'Sentiment', 'Unemployment'],
# python --data=inflatie-bnr-ro.xlsx --model=svr --train-start=2006:Q1 --train-end=2022:Q4 --test-start=2021:Q3  -features=['HICP'] --experiment_name=ROunivariate
# python --data=inflatie-bnr-ro.xlsx --model=svr --train-start=2006:Q1 --train-end=2022:Q4 --test-start=2021:Q3  -features=['HICP', 'Sentiment'] --experiment_name=ROmultivaritateCPI_ESI
# python --data=inflatie-bnr-ro.xlsx --model=svr --train-start=2006:Q1 --train-end=2022:Q4 --test-start=2021:Q3  -features=['HICP', 'Unemployment'] --experiment_name=ROmultivaritateCPI_UNEMPL
# python --data=inflatie-bnr-ro.xlsx --model=svr --train-start=2006:Q1 --train-end=2022:Q4 --test-start=2021:Q3  -features=['HICP', 'Sentiment', 'Unemployment'] --experiment_name=ROmultivaritateCPI_ESI_UNEMPL
# python --data=inflatie-bnr-ro.xlsx --model=rf --train-start=2006:Q1 --train-end=2022:Q4 --test-start=2021:Q3  -features=['HICP'] --experiment_name=ROunivariate
# python --data=inflatie-bnr-ro.xlsx --model=rf --train-start=2006:Q1 --train-end=2022:Q4 --test-start=2021:Q3  -features=['HICP', 'Sentiment'] -experiment_name=ROmultivaritateCPI_ESI
# python --data=inflatie-bnr-ro.xlsx --model=rf --train-start=2006:Q1 --train-end=2022:Q4 --test-start=2021:Q3  -features=['HICP', 'Unemployment'] -experiment_name=ROmultivaritateCPI_UNEMPL
# python --data=inflatie-bnr-ro.xlsx --model=rf --train-start=2006:Q1 --train-end=2022:Q4 --test-start=2021:Q3  -features=['HICP', 'Sentiment', 'Unemployment'] --experiment_name=ROmultivaritateCPI_ESI_UNEMPL
# python --data=inflatie-bnr-ro.xlsx --model=lstm --train-start=2006:Q1 --train-end=2022:Q4 --test-start=2021:Q3 -features=['HICP'] --experiment_name=ROunivariate
# python --data=inflatie-bnr-ro.xlsx --model=lstm --train-start=2006:Q1 --train-end=2022:Q4 --test-start=2021:Q3 -features=['HICP', 'Sentiment'] --experiment_name=ROmultivaritateCPI_ESI
# python --data=inflatie-bnr-ro.xlsx --model=lstm --train-start=2006:Q1 --train-end=2022:Q4 --test-start=2021:Q3 -features=['HICP', 'Unemployment'] --experiment_name=ROmultivaritateCPI_UNEMPL
# python --data=inflatie-bnr-ro.xlsx --model=lstm --train-start=2006:Q1 --train-end=2022:Q4 --test-start=2021:Q3 -features=['HICP', 'Sentiment', 'Unemployment'] --experiment_name=ROmultivaritateCPI_ESI_UNEMPL
