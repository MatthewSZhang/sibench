#!/bin/bash

datasets=("EMPS" "CED" "WienerHammerBenchMark" "Silverbox" "F16" "ParWH" "Cascaded_Tanks")

mkdir -p logs

for data in "${datasets[@]}"; do
    echo "Starting process for: $data"
    uv run opt-pysindy --data "$data" > "logs/pysindy_${data}.log" 2>&1 &
    uv run opt-fastcan --data "$data" > "logs/fastcan_${data}.log" 2>&1 &
    uv run opt-nixtla --data "$data" > "logs/nixtla_${data}.log" 2>&1 &
    uv run opt-sktime --data "$data" > "logs/sktime_${data}.log" 2>&1 &
done

wait

echo "All processes completed. Check the 'logs' folder for results."