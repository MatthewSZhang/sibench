import pandas as pd
import time
import click
import tracemalloc
from ._fastcan import test_opt as fastcan_test_opt
from ._pysindy import test_opt as pysindy_test_opt
from ._nixtla import test_opt as nixtla_test_opt



datasets = [
    "EMPS",
    "CED",
    "WienerHammerBenchMark",
    "Cascaded_Tanks",
]

metric_name = "NRMSE"

@click.command()
@click.option("--db-path", default="results", help="Path to results directory")
def results(db_path: str):
    fastcan_results = []
    pysindy_results = []
    nixtla_results = []
    fastcan_times = []
    pysindy_times = []
    nixtla_times = []
    fastcan_peak_mem = []
    pysindy_peak_mem = []
    nixtla_peak_mem = []
    for data in datasets:
        print(f"Running fastcan on {data}...")
        tracemalloc.start()
        start_time = time.time()
        fastcan_results.append(fastcan_test_opt(data, db_path, return_metric=metric_name))
        fastcan_times.append(time.time() - start_time)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        fastcan_peak_mem.append(peak / 2**20)

        print(f"Running pysindy on {data}...")
        tracemalloc.start()
        start_time = time.time()
        pysindy_results.append(pysindy_test_opt(data, db_path, return_metric=metric_name))
        pysindy_times.append(time.time() - start_time)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pysindy_peak_mem.append(peak / 2**20)

        print(f"Running nixtla on {data}...")
        tracemalloc.start()
        start_time = time.time()
        nixtla_results.append(nixtla_test_opt(data, db_path, return_metric=metric_name))
        nixtla_times.append(time.time() - start_time)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        nixtla_peak_mem.append(peak / 2**20)

    results = pd.DataFrame({
        "Data": datasets,
        "fastcan": fastcan_results,
        "fastcan_time": fastcan_times,
        "fastcan_mem_mb": fastcan_peak_mem,
        "pysindy": pysindy_results,
        "pysindy_time": pysindy_times,
        "pysindy_mem_mb": pysindy_peak_mem,
        "nixtla": nixtla_results,
        "nixtla_time": nixtla_times,
        "nixtla_mem_mb": nixtla_peak_mem,
    })

    results.to_csv("benchmarks_results.csv", index=False)
    return results
