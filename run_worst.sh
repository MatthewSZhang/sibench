#!/bin/bash

datasets=("EMPS" "CED" "WienerHammerBenchMark" "Cascaded_Tanks")

export PYTHONUNBUFFERED=1

mkdir -p logs
mkdir -p results_worst

for data in "${datasets[@]}"; do
    echo "Starting worst case process for: $data"
    
    # FastCAN: n_terms=20, n_lags=10, n_polys=3
    uv run test-fastcan --data "$data" --n-terms 20 --n-lags 10 --n-polys 3 --time-mem "results_worst/fastcan_${data}.csv" > "logs/worst_fastcan_${data}.log" 2>&1 &
    
    # PySINDy: n_degrees=3, n_freqs=10, n_orders=3, threshold=1e-6, alpha=1e-6, rtol=1e-3, atol=1e-3
    uv run test-pysindy --data "$data" --n-degrees 3 --n-freqs 10 --n-orders 3 --threshold 1e-6 --alpha 1e-6 --rtol 1e-3 --atol 1e-3 --time-mem "results_worst/pysindy_${data}.csv" > "logs/worst_pysindy_${data}.log" 2>&1 &
    
    # Nixtla: season_length=50, max_p=5, max_q=5, max_P=1, max_Q=1
    uv run test-nixtla --data "$data" --season-length 50 --max-p 5 --max-q 5 --max-P 1 --max-Q 1 --time-mem "results_worst/nixtla_${data}.csv" > "logs/worst_nixtla_${data}.log" 2>&1 &

done

wait

echo "All processes completed. Check 'results_worst' folder for CSV results and 'logs' for logs."
