#!/bin/bash

datasets=("EMPS" "CED" "WienerHammerBenchMark" "Cascaded_Tanks")

export PYTHONUNBUFFERED=1

mkdir -p logs

for data in "${datasets[@]}"; do
    echo "Starting process for: $data"
    uv run opt-pysindy --data "$data" --n-folds 1 --n-trials 100 > "logs/pysindy_${data}.log" 2>&1 &
    uv run opt-fastcan --data "$data" --n-folds 1 --n-trials 100 > "logs/fastcan_${data}.log" 2>&1 &
done

wait

echo "All processes completed. Check the 'logs' folder for results."