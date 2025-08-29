#!/usr/bin/env python3
"""
evaluate_simulations.py

Full evaluator:
 - reads sim_output/<scenario>/*.csv
 - fits ARDL / RF / SVR (poly) / optional LSTM
 - saves trained models per replication
 - produces recursive forecasts for horizons [1,3,6]
 - computes RMSE/MAE/MAPE per replication-method-horizon
 - aggregates results and saves:
     results/per_replication_metrics.csv
     results/summary_by_scenario.csv
     results/winrate_ci.csv
     results/pairwise_tests.csv
 - produces plots (boxplots & winrate heatmaps)

Usage:
  python evaluate_simulations.py
  python evaluate_simulations.py --sim sim_output --out results --no-lstm --nboot 2000

Dependencies:
  numpy, pandas, scikit-learn, statsmodels, joblib, matplotlib, scipy (for tests).
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from joblib import dump
import pickle
import matplotlib.pyplot as plt
import math
import warnings
from scipy import stats
warnings.filterwarnings("ignore")

# optional tensorflow for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Model
    tf_available = True
except Exception:
    tf_available = False
    print("TensorFlow not available: LSTM will be skipped. To enable LSTM install tensorflow.")

# -----------------------------
# Defaults (should match simulate_cpi.py)
# -----------------------------
N_LAGS = 6
HORIZONS = [1, 3, 6]
ARDL_LAGS = 6
EXOG_LAGS = 6
RF_N_ESTIMATORS = 200
SVR_KERNEL = 'poly'
SVR_DEGREE = 1
SVR_C = 3.0
SVR_EPS = 0.01
LSTM_EPOCHS = 500
LSTM_BATCH = 1
RANDOM_STATE = 12345
BOOTSTRAP_DEFAULT = 2000
# -----------------------------

np.random.seed(RANDOM_STATE)

def safe_mape(y_true, y_pred):
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

# -------------------------
# Feature builders / models
# -------------------------
def build_ardl_features(df, p=ARDL_LAGS, exog_lags=EXOG_LAGS):
    df2 = df.copy().reset_index(drop=True)
    for lag in range(1, p+1):
        df2[f'lnCPI_lag{lag}'] = df2['lnCPI'].shift(lag)
    for lag in range(1, exog_lags+1):
        df2[f'unemp_lag{lag}'] = df2['unemployment'].shift(lag)
        df2[f'sent_lag{lag}'] = df2['sentiment'].shift(lag)
    df2 = df2.dropna().reset_index(drop=True)
    X_cols = [c for c in df2.columns if ('lnCPI_lag' in c) or ('unemp_lag' in c) or ('sent_lag' in c)]
    X = df2[X_cols]
    y = df2['lnCPI']
    return X, y, df2

def fit_ardl_and_forecast(df, train_end_idx, horizons=HORIZONS, p=ARDL_LAGS, exog_lags=EXOG_LAGS):
    max_h = max(horizons)
    X_all, y_all, df_for = build_ardl_features(df, p=p, exog_lags=exog_lags)
    train_date = df.loc[train_end_idx, 'date']
    train_idx_in_df_for = df_for[df_for['date'] <= train_date].index.max()
    if train_idx_in_df_for is None or pd.isna(train_idx_in_df_for):
        train_idx_in_df_for = len(df_for) - 1
    X_train = X_all.loc[:train_idx_in_df_for, :].copy()
    y_train = y_all.loc[:train_idx_in_df_for].copy()
    X_train_const = sm.add_constant(X_train, has_constant='add')
    model = sm.OLS(y_train, X_train_const).fit()

    history = df.copy().reset_index(drop=True)
    forecasts = []
    param_names = list(model.params.index)

    for h in range(1, max_h + 1):
        t_idx = train_end_idx + h
        feat = {}
        for lag in range(1, p+1):
            idx_needed = t_idx - lag
            feat[f'lnCPI_lag{lag}'] = float(history.loc[idx_needed, 'lnCPI'])
        for lag in range(1, exog_lags+1):
            idx_needed = t_idx - lag
            if idx_needed <= train_end_idx:
                val_u = float(history.loc[idx_needed, 'unemployment'])
                val_s = float(history.loc[idx_needed, 'sentiment'])
            else:
                val_u = float(history.loc[train_end_idx, 'unemployment'])
                val_s = float(history.loc[train_end_idx, 'sentiment'])
            feat[f'unemp_lag{lag}'] = val_u
            feat[f'sent_lag{lag}'] = val_s

        xdict = {}
        for name in param_names:
            if name.lower() in ('const', 'intercept', 'constant'):
                xdict[name] = 1.0
            else:
                xdict[name] = float(feat.get(name, 0.0))
        Xp_aligned = pd.DataFrame([xdict], columns=param_names)
        lnC_fore = model.predict(Xp_aligned).iloc[0]

        new_row = history.loc[train_end_idx].copy()
        new_row['date'] = history.loc[train_end_idx, 'date'] + pd.DateOffset(months=h)
        new_row['lnCPI'] = lnC_fore
        new_row['CPI'] = np.exp(lnC_fore)
        history = pd.concat([history, new_row.to_frame().T], ignore_index=True)
        forecasts.append(lnC_fore)

    return np.array(forecasts), model

def make_features_for_ml(df, n_lags=N_LAGS):
    df2 = df.reset_index(drop=True).copy()
    rows = []
    dates = []
    for t in range(n_lags, len(df2)):
        ln_lags = df2['lnCPI'].iloc[t-n_lags:t].values[::-1]
        unemp_lag = df2['unemployment'].iloc[t-1]
        sent_lag = df2['sentiment'].iloc[t-1]
        feat = np.concatenate([ln_lags, [unemp_lag, sent_lag]])
        rows.append(feat)
        dates.append(df2.loc[t, 'date'])
    if len(rows) == 0:
        return np.empty((0, n_lags+2)), np.array([]), []
    X = np.vstack(rows)
    y = df2['lnCPI'].iloc[n_lags:].values
    return X, y, dates

def recursive_rf_forecast(rf, df, train_end_idx, max_h=6, n_lags=N_LAGS):
    history = df.reset_index(drop=True).copy()
    forecasts = []
    for h in range(1, max_h+1):
        last_idx = train_end_idx + (h-1)
        ln_lags = history['lnCPI'].iloc[last_idx-n_lags+1:last_idx+1].values[::-1]
        if len(ln_lags) < n_lags:
            pad = np.full(n_lags - len(ln_lags), history['lnCPI'].iloc[0])
            ln_lags = np.concatenate([ln_lags, pad])[:n_lags]
        unemp = history.loc[train_end_idx, 'unemployment']
        sent = history.loc[train_end_idx, 'sentiment']
        feat = np.concatenate([ln_lags, [unemp, sent]]).reshape(1, -1)
        ln_pred = rf.predict(feat)[0]
        new_row = history.loc[train_end_idx].copy()
        new_row['date'] = history.loc[train_end_idx, 'date'] + pd.DateOffset(months=h)
        new_row['lnCPI'] = ln_pred
        new_row['CPI'] = np.exp(ln_pred)
        history = pd.concat([history, new_row.to_frame().T], ignore_index=True)
        forecasts.append(ln_pred)
    return np.array(forecasts)

def recursive_svr_forecast(svr, scaler, df, train_end_idx, max_h=6, n_lags=N_LAGS):
    history = df.reset_index(drop=True).copy()
    forecasts = []
    for h in range(1, max_h+1):
        last_idx = train_end_idx + (h-1)
        ln_lags = history['lnCPI'].iloc[last_idx-n_lags+1:last_idx+1].values[::-1]
        if len(ln_lags) < n_lags:
            pad = np.full(n_lags - len(ln_lags), history['lnCPI'].iloc[0])
            ln_lags = np.concatenate([ln_lags, pad])[:n_lags]
        unemp = history.loc[train_end_idx, 'unemployment']
        sent = history.loc[train_end_idx, 'sentiment']
        feat = np.concatenate([ln_lags, [unemp, sent]]).reshape(1, -1)
        feat_s = scaler.transform(feat)
        ln_pred = svr.predict(feat_s)[0]
        new_row = history.loc[train_end_idx].copy()
        new_row['date'] = history.loc[train_end_idx, 'date'] + pd.DateOffset(months=h)
        new_row['lnCPI'] = ln_pred
        new_row['CPI'] = np.exp(ln_pred)
        history = pd.concat([history, new_row.to_frame().T], ignore_index=True)
        forecasts.append(ln_pred)
    return np.array(forecasts)

def train_lstm_and_forecast(df, train_end_idx, n_lags=N_LAGS, max_h=6, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, seed=RANDOM_STATE):
    if not tf_available:
        raise ImportError("TensorFlow not available.")
    X_all, y_all, dates = make_features_for_ml(df, n_lags=n_lags)
    train_date = df.loc[train_end_idx, 'date']
    train_idx = max(i for i, d in enumerate(dates) if d <= train_date)
    X_train = X_all[:train_idx+1]
    y_train = y_all[:train_idx+1]
    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler().fit(X_train)
    scalerY = StandardScaler().fit(y_train.reshape(-1,1))
    X_train_s = scalerX.transform(X_train)
    y_train_s = scalerY.transform(y_train.reshape(-1,1))
    seq_train = X_train_s[:, :n_lags].reshape((X_train_s.shape[0], n_lags, 1))
    exog_train = X_train_s[:, n_lags:]
    tf.random.set_seed(seed)
    seq_input = tf.keras.Input(shape=(n_lags, 1), name='seq')
    ex_input = tf.keras.Input(shape=(exog_train.shape[1],), name='exog')
    x = LSTM(512, activation='relu')(seq_input)
    x = tf.keras.layers.Concatenate()([x, ex_input])
    out = Dense(1)(x)
    model = tf.keras.Model([seq_input, ex_input], out)
    model.compile(optimizer='adam', loss='mse')
    model.fit([seq_train, exog_train], y_train_s, epochs=epochs, batch_size=batch_size, verbose=0)
    history = df.reset_index(drop=True).copy()
    forecasts = []
    for h in range(1, max_h+1):
        last_idx = train_end_idx + (h-1)
        seq_vals = history['lnCPI'].iloc[last_idx-n_lags+1:last_idx+1].values
        if len(seq_vals) < n_lags:
            pad = np.full(n_lags - len(seq_vals), history['lnCPI'].iloc[0])
            seq_vals = np.concatenate([seq_vals, pad])[:n_lags]
        feat = np.concatenate([seq_vals[::-1], [history.loc[train_end_idx, 'unemployment'], history.loc[train_end_idx, 'sentiment']]]).reshape(1,-1)
        feat_s = scalerX.transform(feat)
        seq_scaled = feat_s[:, :n_lags].reshape((1, n_lags, 1))
        exog_scaled = feat_s[:, n_lags:]
        yhat_s = model.predict([seq_scaled, exog_scaled], verbose=0)
        yhat = scalerY.inverse_transform(yhat_s.reshape(-1,1)).ravel()[0]
        new_row = history.loc[train_end_idx].copy()
        new_row['date'] = history.loc[train_end_idx, 'date'] + pd.DateOffset(months=h)
        new_row['lnCPI'] = yhat
        new_row['CPI'] = np.exp(yhat)
        history = pd.concat([history, new_row.to_frame().T], ignore_index=True)
        forecasts.append(yhat)
    return np.array(forecasts), model, scalerX, scalerY

# -------------------------
# persistence, plotting and statistics helpers
# -------------------------
def save_model_dir(base_results_dir, scenario, repl_filename):
    rep_id = Path(repl_filename).stem
    model_dir = Path(base_results_dir) / "models" / scenario / rep_id
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir)

def plot_and_save(df_metrics, summary_df, out_dir):
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # boxplot per horizon
    for h in sorted(df_metrics['horizon'].unique()):
        sub = df_metrics[df_metrics['horizon'] == h].dropna(subset=['rmse'])
        if sub.empty:
            continue
        plt.figure(figsize=(8,5))
        methods = sorted(sub['method'].unique())
        data = [sub[sub['method']==m]['rmse'].values for m in methods]
        plt.boxplot(data, labels=methods, notch=True)
        plt.title(f'RMSE by method (horizon={h} months)')
        plt.ylabel('RMSE (lnCPI)')
        plt.tight_layout()
        plt.savefig(plots_dir / f"boxplot_rmse_h{h}.png", dpi=200)
        plt.close()

    # combined boxplot
    try:
        plt.figure(figsize=(10,6))
        methods = sorted(df_metrics['method'].unique())
        data = []
        labels = []
        for h in sorted(df_metrics['horizon'].unique()):
            for m in methods:
                vals = df_metrics[(df_metrics['horizon']==h) & (df_metrics['method']==m)]['rmse'].dropna().values
                data.append(vals)
                labels.append(f"{m}\n(h={h})")
        if len(data) > 0:
            plt.boxplot(data, labels=labels, notch=True)
            plt.xticks(rotation=45, ha='right')
            plt.title("RMSE by method and horizon")
            plt.tight_layout()
            plt.savefig(plots_dir / "boxplot_rmse_by_horizon.png", dpi=200)
            plt.close()
    except Exception as e:
        print("Failed to save combined boxplot:", e)

def plot_winrate_heatmaps(winrate_df, out_dir):
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    # winrate_df expected columns: scenario, horizon, method, win_rate
    for scen in winrate_df['scenario'].unique():
        piv = winrate_df[winrate_df['scenario']==scen].pivot(index='method', columns='horizon', values='win_rate').fillna(0.0)
        if piv.empty:
            continue
        plt.figure(figsize=(6, max(3, 0.6 * len(piv.index))))
        im = plt.imshow(piv.values, aspect='auto', interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04, label='win rate')
        plt.yticks(range(len(piv.index)), piv.index)
        plt.xticks(range(len(piv.columns)), piv.columns)
        plt.title(f'Win rate heatmap — {scen}')
        plt.xlabel('Horizon (months)')
        plt.tight_layout()
        plt.savefig(plots_dir / f"winrate_heatmap_{scen}.png", dpi=200)
        plt.close()

# -------------------------
# Win-rate + bootstrap CI + pairwise tests
# -------------------------
def compute_fractional_winrates(pivot, methods=None, tol=1e-12):
    """
    pivot: DataFrame index=replication_file, columns=method, values=loss (rmse)
    returns: Series indexed by method with fractional win counts (not normalized)
    """
    if methods is None:
        methods = list(pivot.columns)
    wins = pd.Series(0.0, index=methods, dtype=float)
    rep_ids = pivot.index.values
    for r in rep_ids:
        losses = pivot.loc[r, methods]
        if losses.isna().all():
            continue
        minval = losses.min(skipna=True)
        is_min = (losses - minval) <= tol
        k = int(is_min.sum())
        if k == 0:
            continue
        wins[is_min.index] += is_min.astype(float) / float(k)
    return wins, len(rep_ids)

def bootstrap_winrate_ci(pivot, methods=None, nboot=1000, alpha=0.05, random_state=RANDOM_STATE):
    """
    pivot: DataFrame index=replication_file, columns=method, values=loss
    returns: dict method -> (mean_winrate, low, high) (percentile CI)
    """
    if methods is None:
        methods = list(pivot.columns)
    rep_ids = pivot.index.values
    rng = np.random.default_rng(random_state)
    boot_vals = {m: [] for m in methods}
    n_rep = len(rep_ids)
    if n_rep == 0:
        return {m: {'win_rate': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan} for m in methods}
    for b in range(nboot):
        sample_ids = rng.choice(rep_ids, size=n_rep, replace=True)
        pivot_b = pivot.loc[sample_ids]
        wins_b, _ = compute_fractional_winrates(pivot_b, methods=methods)
        winrates_b = wins_b / float(n_rep)
        for m in methods:
            boot_vals[m].append(winrates_b.get(m, 0.0))
    res = {}
    for m in methods:
        arr = np.array(boot_vals[m])
        res[m] = {'win_rate': float(arr.mean()), 'ci_lower': float(np.percentile(arr, 100*alpha/2)), 'ci_upper': float(np.percentile(arr, 100*(1-alpha/2)))}
    return res

def pairwise_tests_for_pivot(pivot, methods=None):
    """
    pivot: DataFrame index=replication_file, columns=method, values=rmse
    returns: list of dicts with pairwise test results for all ordered pairs (A,B)
    fields: scenario, horizon, method_A, method_B, n_pairs, mean_diff (A-B), sd_diff, p_ttest, p_wilcoxon, prop_A_better
    where prop_A_better = fraction of reps where rmse_A < rmse_B (ties excluded from numerator but counted in denominator)
    """
    if methods is None:
        methods = list(pivot.columns)
    results = []
    rep_ids = pivot.index.values
    for i, A in enumerate(methods):
        for j, B in enumerate(methods):
            if A == B:
                continue
            # keep rows where both are not NaN
            pair_df = pivot[[A, B]].dropna(how='any')
            n = len(pair_df)
            if n == 0:
                row = {'method_A': A, 'method_B': B, 'n_pairs': 0, 'mean_diff': np.nan, 'sd_diff': np.nan,
                       'p_ttest': np.nan, 'p_wilcoxon': np.nan, 'prop_A_better': np.nan}
                results.append(row); continue
            diffs = pair_df[A].values - pair_df[B].values  # negative => A better (smaller rmse)
            mean_diff = float(np.mean(diffs))
            sd_diff = float(np.std(diffs, ddof=1)) if n>1 else float(0.0)
            # paired t-test
            try:
                tstat, p_t = stats.ttest_rel(pair_df[A].values, pair_df[B].values, nan_policy='omit')
            except Exception:
                p_t = np.nan
            # wilcoxon (requires at least one non-zero diff and n>=1)
            try:
                # scipy's wilcoxon requires n>0 and not all zeros
                if n >= 1 and np.any(np.abs(diffs) > 1e-12):
                    wstat, p_w = stats.wilcoxon(pair_df[A].values, pair_df[B].values, zero_method='wilcox', alternative='two-sided')
                else:
                    p_w = np.nan
            except Exception:
                p_w = np.nan
            # proportion A better (strict)
            prop_A_better = float(np.mean(pair_df[A].values < pair_df[B].values))
            row = {'method_A': A, 'method_B': B, 'n_pairs': n, 'mean_diff': mean_diff, 'sd_diff': sd_diff,
                   'p_ttest': float(p_t) if (p_t is not None and not np.isnan(p_t)) else np.nan,
                   'p_wilcoxon': float(p_w) if (p_w is not None and not np.isnan(p_w)) else np.nan,
                   'prop_A_better': prop_A_better}
            results.append(row)
    return results

# -------------------------
# Main evaluation loop
# -------------------------
def evaluate(sim_root="sim_output", result_dir="results", no_lstm=False, nboot=BOOTSTRAP_DEFAULT):
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    per_rep_rows = []
    scenarios = sorted([d for d in os.listdir(sim_root) if os.path.isdir(os.path.join(sim_root, d))])
    if not scenarios:
        raise FileNotFoundError(f"No scenario subdirectories found in {sim_root} -- run simulate_cpi.py first.")
    for scen in scenarios:
        scen_dir = os.path.join(sim_root, scen)
        print("Processing scenario:", scen)
        csvs = sorted(glob.glob(os.path.join(scen_dir, f"{scen}_rep*.csv")))
        if not csvs:
            print("  no replication files found for", scen)
            continue
        for csvf in csvs:
            repl_name = os.path.basename(csvf)
            try:
                df = pd.read_csv(csvf, parse_dates=['date'])
            except Exception as e:
                print("Failed to read", csvf, ":", e)
                continue
            df = df.sort_values('date').reset_index(drop=True)
            n = len(df)
            if n < max(HORIZONS) + N_LAGS + 10:
                print("  skipping too-short series:", csvf)
                continue
            train_end_idx = n - max(HORIZONS) - 1
            max_h = max(HORIZONS)
            true_future = [df.loc[train_end_idx + h, 'lnCPI'] for h in range(1, max_h+1)]
            true_future = np.array(true_future)

            model_dir = save_model_dir(result_dir, scen, repl_name)

            # ARDL
            try:
                ardl_fore, ardl_model = fit_ardl_and_forecast(df, train_end_idx, horizons=HORIZONS, p=ARDL_LAGS, exog_lags=EXOG_LAGS)
                ardl_path = os.path.join(model_dir, "ARDL_results.pkl")
                try:
                    ardl_model.save(ardl_path)
                except Exception:
                    with open(ardl_path, "wb") as f:
                        pickle.dump({'params': ardl_model.params, 'bse': ardl_model.bse, 'rsquared': ardl_model.rsquared}, f)
            except Exception as e:
                print("ARDL failed for", csvf, ":", e)
                ardl_fore = np.full(max_h, np.nan)

            # Prepare ML features
            try:
                X_all, y_all, dates = make_features_for_ml(df, n_lags=N_LAGS)
                train_date = df.loc[train_end_idx, 'date']
                train_idx_feat = max(i for i, d in enumerate(dates) if d <= train_date)
                X_train = X_all[:train_idx_feat+1]
                y_train = y_all[:train_idx_feat+1]
            except Exception as e:
                print("Feature construction failed for", csvf, ":", e)
                X_train = np.empty((0, N_LAGS+2))
                y_train = np.empty((0,))

            # RF
            try:
                rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, min_samples_split=2, max_depth=10, max_features=0.5, n_jobs=1)
                rf.fit(X_train, y_train)
                rf_fore = recursive_rf_forecast(rf, df, train_end_idx, max_h=max_h, n_lags=N_LAGS)
                dump(rf, os.path.join(model_dir, "RF.joblib"))
            except Exception as e:
                print("RF failed for", csvf, ":", e)
                rf_fore = np.full(max_h, np.nan)

            # SVR (polynomial)
            try:
                scaler_svr = StandardScaler()
                X_train_s = scaler_svr.fit_transform(X_train)
                svr = SVR(kernel=SVR_KERNEL, degree=SVR_DEGREE, C=SVR_C, coef0= 0.01, epsilon=SVR_EPS)
                svr.fit(X_train_s, y_train)
                svr_fore = recursive_svr_forecast(svr, scaler_svr, df, train_end_idx, max_h=max_h, n_lags=N_LAGS)
                dump(scaler_svr, os.path.join(model_dir, "SVR_scaler.joblib"))
                dump(svr, os.path.join(model_dir, "SVR.joblib"))
            except Exception as e:
                print("SVR failed for", csvf, ":", e)
                svr_fore = np.full(max_h, np.nan)

            # LSTM (optional) with robust saving
            if tf_available and (not no_lstm):
                try:
                    lstm_fore, lstm_model, lstm_scalerX, lstm_scalerY = train_lstm_and_forecast(df, train_end_idx, n_lags=N_LAGS, max_h=max_h, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH)
                    lstm_saved = False
                    lstm_path = os.path.join(model_dir, "LSTM_model")
                    try:
                        lstm_model.save(lstm_path, save_format='tf')
                        lstm_saved = True
                    except Exception as e_sm:
                        try:
                            weights_path = os.path.join(model_dir, "LSTM.weights.h5")
                            lstm_model.save_weights(weights_path)
                            lstm_saved = True
                        except Exception as e_w:
                            try:
                                weights_path2 = os.path.join(model_dir, "LSTM_weights.h5")
                                lstm_model.save_weights(weights_path2)
                                lstm_saved = True
                            except Exception as e_w2:
                                print(f"Failed to save LSTM (SavedModel: {e_sm}; weights: {e_w}; alt: {e_w2})")
                    try:
                        dump(lstm_scalerX, os.path.join(model_dir, "LSTM_scalerX.joblib"))
                        dump(lstm_scalerY, os.path.join(model_dir, "LSTM_scalerY.joblib"))
                    except Exception as e_scaler:
                        print("Warning: failed to save LSTM scalers:", e_scaler)
                    if not lstm_saved:
                        print(f"Warning: LSTM model could not be saved for {repl_name}; forecasts still used.")
                except Exception as e:
                    print("LSTM failed for", csvf, ":", e)
                    lstm_fore = np.full(max_h, np.nan)
            else:
                lstm_fore = np.full(max_h, np.nan)

            # record metrics
            for method_name, preds in [('ARDL', ardl_fore), ('RF', rf_fore), ('SVR', svr_fore), ('LSTM', lstm_fore)]:
                for h_idx, h in enumerate(range(1, max_h+1)):
                    if h in HORIZONS:
                        y_true = true_future[h_idx]
                        y_pred = preds[h_idx] if len(preds) >= h_idx+1 else np.nan
                        if np.isnan(y_pred):
                            rmse = np.nan; mae = np.nan; mape = np.nan
                        else:
                            rmse = math.sqrt((y_true - y_pred)**2)
                            mae = abs(y_true - y_pred)
                            mape = safe_mape(np.array([y_true]), np.array([y_pred]))
                        per_rep_rows.append({
                            'scenario': scen,
                            'replication_file': os.path.basename(csvf),
                            'method': method_name,
                            'horizon': h,
                            'rmse': rmse,
                            'mae': mae,
                            'mape': mape
                        })

    # Save per-rep metrics
    df_metrics = pd.DataFrame(per_rep_rows)
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(os.path.join(result_dir, "per_replication_metrics.csv"), index=False)

    # Aggregate summary (mean +/- sd)
    grouped = df_metrics.groupby(['scenario','method','horizon'])
    summary = grouped.agg(n=('rmse','count'),
                          mean_rmse=('rmse','mean'),
                          sd_rmse=('rmse','std'),
                          mean_mae=('mae','mean'),
                          sd_mae=('mae','std'),
                          mean_mape=('mape','mean'),
                          sd_mape=('mape','std')).reset_index()

    # Compute fractional win-rates and bootstrap CI, and pairwise tests
    winrate_rows = []
    pairwise_rows = []
    scenarios_unique = df_metrics['scenario'].unique()
    for scen in scenarios_unique:
        for h in sorted(df_metrics[horizon_colname := 'horizon'].unique()):
            sub = df_metrics[(df_metrics['scenario']==scen) & (df_metrics['horizon']==h)]
            if sub.empty:
                continue
            pivot = sub.pivot_table(index='replication_file', columns='method', values='rmse')
            # fractional wins (counts)
            wins_counts, n_rep = compute_fractional_winrates(pivot)
            if n_rep == 0:
                continue
            winrates = wins_counts / float(n_rep)
            # bootstrap CI
            bootres = bootstrap_winrate_ci(pivot, methods=list(pivot.columns), nboot=nboot)
            for m in sorted(pivot.columns):
                row = {
                    'scenario': scen,
                    'horizon': h,
                    'method': m,
                    'win_rate': float(winrates.get(m, 0.0)),
                    'win_ci_lower': float(bootres[m]['ci_lower']),
                    'win_ci_upper': float(bootres[m]['ci_upper']),
                    'n_replications_used': int(n_rep)
                }
                winrate_rows.append(row)
            # pairwise tests
            pairwise = pairwise_tests_for_pivot(pivot, methods=list(pivot.columns))
            for p in pairwise:
                p.update({'scenario': scen, 'horizon': h})
                pairwise_rows.append(p)

    df_winrate = pd.DataFrame(winrate_rows)
    df_pairwise = pd.DataFrame(pairwise_rows)

    # Merge winrate into summary_by_scenario (optional)
    if not df_winrate.empty:
        # pivot win rates for easier viewing
        # but also merge: choose average win_rate per method/horizon/scenario
        win_agg = df_winrate.groupby(['scenario','method','horizon'])[['win_rate','win_ci_lower','win_ci_upper','n_replications_used']].first().reset_index()
        summary = summary.merge(win_agg, on=['scenario','method','horizon'], how='left')

    # Save outputs
    summary.to_csv(os.path.join(result_dir, "summary_by_scenario.csv"), index=False)
    df_winrate.to_csv(os.path.join(result_dir, "winrate_ci.csv"), index=False)
    df_pairwise.to_csv(os.path.join(result_dir, "pairwise_tests.csv"), index=False)

    # Markdown summary for manuscript (lightweight)
    md_lines = []
    for scen, sub in summary.groupby('scenario'):
        md_lines.append(f"## Scenario: {scen}\n")
        md_lines.append("| Method | Horizon | mean RMSE | sd RMSE | mean MAPE (%) | win rate (95% CI) |")
        md_lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in sub.sort_values(['method','horizon']).iterrows():
            wr = f"{row.get('win_rate', np.nan):.2f}"
            cil = row.get('win_ci_lower', np.nan)
            ciu = row.get('win_ci_upper', np.nan)
            if not np.isnan(cil) and not np.isnan(ciu):
                wr = f"{row.get('win_rate', np.nan):.2f} ({cil:.2f}-{ciu:.2f})"
            md_lines.append(f"| {row['method']} | {int(row['horizon'])} | {row['mean_rmse']:.4f} | {row['sd_rmse']:.4f} | {row['mean_mape']:.2f}% | {wr} |")
        md_lines.append("\n")
    with open(os.path.join(result_dir, "summary_markdown.md"), "w") as f:
        f.write("\n".join(md_lines))

    # plots
    try:
        plot_and_save(df_metrics, summary, result_dir)
        if not df_winrate.empty:
            plot_winrate_heatmaps(df_winrate, result_dir)
    except Exception as e:
        print("Plotting failed:", e)

    print("Evaluation finished. Results placed in:", result_dir)
    print("Wrote: per_replication_metrics.csv, summary_by_scenario.csv, winrate_ci.csv, pairwise_tests.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, default="sim_output", help="Simulation root directory")
    parser.add_argument("--out", type=str, default="results", help="Results directory")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM even if TF present")
    parser.add_argument("--nboot", type=int, default=BOOTSTRAP_DEFAULT, help="Bootstrap replications for winrate CI")
    args = parser.parse_args()
    evaluate(sim_root=args.sim, result_dir=args.out, no_lstm=args.no_lstm, nboot=args.nboot)