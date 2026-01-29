import pandas as pd
from ._fastcan import test_opt as fastcan_test_opt
from ._pysindy import test_opt as pysindy_test_opt

results_path = "/Users/sikai/Documents/code/github/sibench/0128 results"


datasets = [
    "EMPS",
    "CED",
    "WienerHammerBenchMark",
    "Cascaded_Tanks",
]

def run_all_benchmarks(results_path: str):
    fastcan_results = []
    pysindy_results = []
    for data in datasets:
        print(f"Running fastcan on {data}...")
        fastcan_results.append(fastcan_test_opt(data, results_path))
        print(f"Running pysindy on {data}...")
        pysindy_results.append(pysindy_test_opt(data, results_path))
    return pd.DataFrame({
        "Dataset": datasets,
        "fastcan RMSE": fastcan_results,
        "pysindy RMSE": pysindy_results,
    })


