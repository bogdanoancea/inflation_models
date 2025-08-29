#!/usr/bin/env python3
"""
simulate_cpi.py  (N_LAGS = 6, safe breaks)

Generates simulated monthly CPI series (ln(CPI) process) under configurable scenarios.
Aligned to N_LAGS = 6 and MAX_HORIZON = 6. Writes per-rep CSVs under sim_output/<scenario>/.

Usage:
  python simulate_cpi.py            # default R=200, T=120
  python simulate_cpi.py --reps 100 --T 180
"""
import os
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# ----- CONFIG / constants -----
N_LAGS = 6                # number of lags your evaluation pipeline uses
MAX_HORIZON = 6           # maximum forecast horizon in months (we evaluate 1,3,6)
BUFFER_AFTER_HOLDOUT = 12 # additional buffer to ensure enough training data
DEFAULT_T = 240          # default series length (months)
DEFAULT_R = 200           # default replications per scenario
OUT_ROOT = "sim_output"
# -------------------------------

def ar1_process(T, phi, sigma, mu=0.0, seed=None):
    rng = np.random.default_rng(seed)
    eps = rng.normal(scale=sigma, size=T)
    x = np.empty(T)
    if abs(phi) < 1:
        x[0] = mu + eps[0] / (1 - phi)
    else:
        x[0] = mu + eps[0]
    for t in range(1, T):
        x[t] = mu + phi * x[t-1] + eps[t]
    return x

def choose_safe_break(T, candidate_break):
    min_break = N_LAGS + 12
    max_break = max(min_break, T - MAX_HORIZON - 6)
    if max_break <= min_break:
        return None
    if candidate_break is None:
        br = T // 2
    else:
        br = int(candidate_break)
    if br < min_break:
        br = min_break
    if br > max_break:
        br = max_break
    return int(br)

def simulate_one(
    T=DEFAULT_T,
    start_date='2006-01-01',
    phi=0.95,
    mu=0.02,
    beta_u=-0.05,
    beta_s=0.10,
    gamma_inter=0.0,
    phi_u=0.6, sigma_u=0.5,
    phi_s=0.7, sigma_s=0.8,
    ma_theta=0.0,
    sigma_eps=0.5,
    nonlin_threshold=None,
    nonlin_bonus=0.3,
    break_at=None,
    break_shift=0.0,
    obs_noise_sd=0.0,
    missing_rate=0.0,
    seed=None
):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=T, freq='M')
    u = ar1_process(T, phi_u, sigma_u, mu=5.0, seed=(seed and seed+1))
    s = ar1_process(T, phi_s, sigma_s, mu=0.0, seed=(seed and seed+2))

    if ma_theta == 0.0:
        eps = rng.normal(scale=sigma_eps, size=T)
    else:
        w = rng.normal(scale=sigma_eps, size=T)
        eps = np.empty(T)
        eps[0] = w[0]
        for t in range(1, T):
            eps[t] = w[t] + ma_theta * w[t-1]

    lnC = np.empty(T)
    lnC[0] = mu / (1 - phi) if abs(phi) < 1 else 0.0

    safe_break = choose_safe_break(T, break_at)
    for t in range(1, T):
        nonlin_term = 0.0
        if (nonlin_threshold is not None) and (s[t-1] > nonlin_threshold):
            nonlin_term = nonlin_bonus * (s[t-1] - nonlin_threshold)
        interaction = gamma_inter * u[t-1] * s[t-1]
        lnC[t] = (mu
                  + phi * lnC[t-1]
                  + beta_u * u[t-1]
                  + beta_s * s[t-1]
                  + interaction
                  + nonlin_term
                  + eps[t])
        if safe_break is not None and t >= safe_break:
            lnC[t] += break_shift

    CPI = np.exp(lnC)
    if obs_noise_sd > 0:
        CPI = CPI * np.exp(rng.normal(scale=obs_noise_sd, size=T))
        lnC = np.log(CPI)

    df = pd.DataFrame({
        'date': dates,
        'lnCPI': lnC,
        'CPI': CPI,
        'unemployment': u,
        'sentiment': s
    })
    df['CPI_pct_month'] = 100 * df['lnCPI'].diff()
    df['CPI_pct_12m'] = 100 * (df['lnCPI'] - df['lnCPI'].shift(12))
    df['regime_flag'] = ((df['sentiment'] > (nonlin_threshold if nonlin_threshold is not None else 1e9))).astype(int)

    if missing_rate > 0:
        nmiss = int(np.floor(missing_rate * T))
        if nmiss > 0:
            miss_idx = rng.choice(T, size=nmiss, replace=False)
            df.loc[miss_idx, ['CPI', 'lnCPI', 'CPI_pct_month', 'CPI_pct_12m']] = pd.NA

    return df, safe_break

def run_simulations_and_save(out_dir=OUT_ROOT, scenarios=None, R=DEFAULT_R, T=DEFAULT_T, seed_base=12345):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    min_length = N_LAGS + MAX_HORIZON + BUFFER_AFTER_HOLDOUT
    if T < min_length:
        print(f"WARNING: requested T={T} too short. Adjusting T -> {min_length}.")
        T = min_length

    if scenarios is None:
        scenarios = [
            dict(name='baseline_linear', phi=0.95, beta_u=-0.05, beta_s=0.10, ma_theta=0.0,
                 nonlin_threshold=None, break_at=None, break_shift=0.0),
            dict(name='high_persistence', phi=0.99, beta_u=-0.05, beta_s=0.10, ma_theta=0.0,
                 nonlin_threshold=None, break_at=None, break_shift=0.0),
            dict(name='nonlinear_regime', phi=0.90, beta_u=-0.03, beta_s=0.12, ma_theta=0.0,
                 nonlin_threshold=0.5, nonlin_bonus=0.4, gamma_inter=0.01, break_at=None, break_shift=0.0),
            dict(name='structural_break', phi=0.95, beta_u=-0.05, beta_s=0.10, ma_theta=0.0,
                 break_at=None, break_shift=0.08),
            dict(name='ma_noninvertible', phi=0.95, beta_u=-0.05, beta_s=0.10, ma_theta=-1.2,
                 nonlin_threshold=None, break_at=None, break_shift=0.0)
        ]

    metadata = {'N_LAGS': N_LAGS, 'MAX_HORIZON': MAX_HORIZON, 'requested_T': T, 'scenarios': []}

    for scen in scenarios:
        scen_name = scen.get('name', 'scenario')
        scen_dir = Path(out_dir) / scen_name
        scen_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running scenario: {scen_name} (R={R}, T={T})")
        scen_meta = scen.copy()
        scen_meta['T'] = T
        rep_safe_breaks = []
        for r in range(R):
            seed = seed_base + r
            # filter out non-simulation keys (e.g., 'name') before passing to simulate_one
            scen_params = {k: v for k, v in scen.items() if k != 'name'}
            # call simulate_one with explicit T and seed and scenario parameters
            df, safe_break = simulate_one(T=T, seed=seed, **scen_params)
            rep_safe_breaks.append(safe_break)
            fname = scen_dir / f"{scen_name}_rep{r+1:04d}.csv"
            df.to_csv(fname, index=False, date_format='%Y-%m-%d')
        scen_meta['safe_break'] = rep_safe_breaks[0]
        scen_meta['n_replications'] = R
        metadata['scenarios'].append(scen_meta)
        df.head(40).to_csv(scen_dir / f"{scen_name}_example_head.csv", index=False)

    with open(Path(out_dir) / "scenarios_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=int)

    print("Simulation finished. Files placed in:", out_dir)
    print("Metadata written to:", Path(out_dir) / "scenarios_metadata.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=DEFAULT_R, help="Replications per scenario")
    parser.add_argument("--T", type=int, default=DEFAULT_T, help="Months per replication (series length)")
    parser.add_argument("--out", type=str, default=OUT_ROOT, help="Output root directory")
    parser.add_argument("--seed", type=int, default=12345, help="Seed base")
    args = parser.parse_args()
    run_simulations_and_save(out_dir=args.out, R=args.reps, T=args.T, seed_base=args.seed)