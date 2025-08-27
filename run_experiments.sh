#!/usr/bin/env bash
# run_experiments.sh
# Explicit sequential runner for your 12 experiments (no DATA/TRAIN_START env vars).
# Edit DATA_FILE, SCRIPT and dates below if you need different values.

set -euo pipefail

PYTHON=python
SCRIPT=svm-rf_integrated.py
DATA_FILE=dateCEE-hicp.xlsx
TRAIN_START=2006-01
TRAIN_END=2024-12
TEST_START=2024-07

mkdir -p runs_logs

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
# SVR experiments (multicountry)
# ---------------------------
run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=svr --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --multicountry -features=\"['CPI']\" --experiment_name=univariate" "runs_logs/svr_univariate.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=svr --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --multicountry -features=\"['CPI','ESI']\" --experiment_name=multivaritateCPI_ESI" "runs_logs/svr_multivar_CPI_ESI.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=svr --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --multicountry -features=\"['CPI','UNEMPL']\" --experiment_name=multivaritateCPI_UNEMPL" "runs_logs/svr_multivar_CPI_UNEMPL.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=svr --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --multicountry -features=\"['CPI','ESI','UNEMPL']\" --experiment_name=multivaritateCPI_ESI_UNEMPL" "runs_logs/svr_multivar_CPI_ESI_UNEMPL.log"

# ---------------------------
# RF experiments (multicountry)
# ---------------------------
run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=rf --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --multicountry -features=\"['CPI']\" --experiment_name=univariate" "runs_logs/rf_univariate.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=rf --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --multicountry -features=\"['CPI','ESI']\" --experiment_name=multivaritateCPI_ESI" "runs_logs/rf_multivar_CPI_ESI.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=rf --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --multicountry -features=\"['CPI','UNEMPL']\" --experiment_name=multivaritateCPI_UNEMPL" "runs_logs/rf_multivar_CPI_UNEMPL.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=rf --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --multicountry -features=\"['CPI','ESI','UNEMPL']\" --experiment_name=multivaritateCPI_ESI_UNEMPL" "runs_logs/rf_multivar_CPI_ESI_UNEMPL.log"

# ---------------------------
# LSTM experiments (multicountry)
# ---------------------------
run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=lstm --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --multicountry -features=\"['CPI']\" --experiment_name=univariate" "runs_logs/lstm_univariate.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=lstm --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --multicountry -features=\"['CPI','ESI']\" --experiment_name=multivaritateCPI_ESI" "runs_logs/lstm_multivar_CPI_ESI.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=lstm --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --multicountry -features=\"['CPI','UNEMPL']\" --experiment_name=multivaritateCPI_UNEMPL" "runs_logs/lstm_multivar_CPI_UNEMPL.log"

run_cmd "$PYTHON $SCRIPT --data=$DATA_FILE --model=lstm --train-start=$TRAIN_START --train-end=$TRAIN_END --test-start=$TEST_START --multicountry -features=\"['CPI','ESI','UNEMPL']\" --experiment_name=multivaritateCPI_ESI_UNEMPL" "runs_logs/lstm_multivar_CPI_ESI_UNEMPL.log"

echo "All experiments finished."