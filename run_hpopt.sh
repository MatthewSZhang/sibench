#!/bin/bash

source .venv/bin/activate

datasets=("EMPS" "CED" "WienerHammerBenchMark" "Silverbox" "F16" "ParWH" "Cascaded_Tanks")

mkdir -p logs

for data in "${datasets[@]}"; do
    echo "Starting process for: $data"
    opt-pysindy --data "$data" > "logs/pysindy_${data}.log" 2>&1 &
    opt-fastcan --data "$data" > "logs/fastcan_${data}.log" 2>&1 &
done

wait

echo "All processes completed. Check the 'logs' folder for results."