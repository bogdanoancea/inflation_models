#!/usr/bin/env bash
# run_experiments.sh
# Explicit sequential runner for your 12 experiments (no DATA/TRAIN_START env vars).
# Edit DATA_FILE, SCRIPT and dates below if you need different values.

set -euo pipefail

PYTHON=python
SCRIPT=svm-rf-lstm-integrated.py
DATA_FILE=inflatie-bnr-ro.xlsx
TRAIN_START=2006:Q1
TRAIN_END=2022:Q4
TEST_START=2021:Q3

mkdir -p runs_logs_RO

run_cmd () {
    local cmd="$1"; local logfile="$2"
    echo
    echo "-------------------------------------------------------------"
    echo "RUN: $cmd"
    echo "LOG: $logfile"
    echo "-------------------------------------------------------------"
    eval "$cmd" 2>&1 | tee "$logfile"
    echo "Exit status: ${PIPESTATUS[0]}"
    echo "-------------------------------------------------------------"
    echo
}

# ---------------------------
# SVR experiments RO
# ---------------------------
run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=svr --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START  --features=\"['HICP']\" --experiment_name=ROunivariate" "runs_logs_RO/svr_univariate.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=svr --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START  --features=\"['HICP','Sentiment']\" --experiment_name=ROmultivariateHICPSent" "runs_logs_RO/svr_multivar_HICP_Sent.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=svr --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START  --features=\"['HICP','Unemployment']\" --experiment_name=ROmultivariateHICPUnempl" "runs_logs_RO/svr_multivar_HICP_Unempl.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=svr --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --features=\"['HICP','Sentiment','Unemployment']\" --experiment_name=ROmultivariateHICP_Sent_Uempl" "runs_logs_RO/svr_multivar_HICP_Sent_Unempl.log"

# ---------------------------
# RF experiments RO
# ---------------------------
run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=rf --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --features=\"['HICP']\" --experiment_name=ROunivariate" "runs_logs_RO/rf_univariate.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=rf --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --features=\"['HICP','Sentiment']\" --experiment_name=ROmultivariateHICP_Sent" "runs_logs_RO/rf_multivar_HICP_Sent.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=rf --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --features=\"['HICP','Unemployment']\" --experiment_name=ROmultivartateHICP_Unempl" "runs_logs_RO/rf_multivar_HICP_Unempl.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=rf --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --features=\"['HICP','Sentiment','Unemployment']\" --experiment_name=multivariateHICP_Sent_Unempl" "runs_logs_RO/rf_multivar_HICP_Sent_Unempl.log"

# ---------------------------
# LSTM experiments RO
# ---------------------------
run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=lstm --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --features=\"['HICP']\" --experiment_name=ROunivariate" "runs_logs_RO/lstm_univariate.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=lstm --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --features=\"['HICP','Sentiment']\" --experiment_name=ROmultivaritateHICP_Sent" "runs_logs_RO/lstm_multivar_HICP_Sent.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=lstm --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --features=\"['HICP','Unemployment']\" --experiment_name=ROmultivaritateHICP_Unempl" "runs_logs_RO/lstm_multivar_HICP_Unempl.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=lstm --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --features=\"['HICP','Sentiment','Unemployment']\" --experiment_name=ROmultivaritateHICP_Sent_Unempl" "runs_logs_RO/lstm_multivar_HICP_Sent_Unempl.log"

echo "All experiments finished."
