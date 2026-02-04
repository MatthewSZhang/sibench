import pandas as pd
import time
import click
import tracemalloc
from ._fastcan import test_opt as fastcan_test_opt
from ._pysindy import test_opt as pysindy_test_opt
from ._nixtla import test_opt as nixtla_test_opt
from ._fastcan import test as fastcan_test
from ._pysindy import test as pysindy_test
from ._nixtla import test as nixtla_test


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
    for data in datasets:
        print(f"Running fastcan on {data}...")
        fastcan_results.append(
            fastcan_test_opt(data, db_path, return_metric=metric_name)
        )

        print(f"Running pysindy on {data}...")
        pysindy_results.append(
            pysindy_test_opt(data, db_path, return_metric=metric_name)
        )

        print(f"Running nixtla on {data}...")
        nixtla_results.append(nixtla_test_opt(data, db_path, return_metric=metric_name))

    results = pd.DataFrame(
        {
            "Data": datasets,
            "fastcan": fastcan_results,
            "pysindy": pysindy_results,
            "nixtla": nixtla_results,
        }
    )

    results.to_csv("benchmarks_results.csv", index=False)
    return results


@click.command()
def worst_case():
    """
    Check training time and memory usage in worst-case
    parameters (largest search space / complexity).
    """
    fastcan_times = []
    pysindy_times = []
    nixtla_times = []
    fastcan_peak_mem = []
    pysindy_peak_mem = []
    nixtla_peak_mem = []

    # Worst case parameters
    # FastCAN: n_terms=20, n_lags=10, n_polys=3
    # PySINDy: n_degrees=3, n_freqs=10, n_orders=3, threshold=1e-6, alpha=1e-6
    # Nixtla: season_length=50, max_p=5, max_q=5

    for data in datasets:
        print(f"Running fastcan (worst case) on {data}...")
        tracemalloc.start()
        start_time = time.time()
        fastcan_test(data, n_terms=20, n_lags=10, n_polys=3, return_metric=metric_name)
        fastcan_times.append(time.time() - start_time)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        fastcan_peak_mem.append(peak / 2**20)

        print(f"Running pysindy (worst case) on {data}...")
        tracemalloc.start()
        start_time = time.time()
        pysindy_test(
            data,
            n_degrees=3,
            n_freqs=10,
            n_orders=3,
            threshold=1e-6,
            alpha=1e-6,
            rtol=1e-6,
            atol=1e-6,
            return_metric=metric_name,
        )
        pysindy_times.append(time.time() - start_time)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pysindy_peak_mem.append(peak / 2**20)

        print(f"Running nixtla (worst case) on {data}...")
        tracemalloc.start()
        start_time = time.time()
        nixtla_test(
            data,
            season_length=50,
            max_p=5,
            max_q=5,
            max_P=1,
            max_Q=1,
            return_metric=metric_name,
        )
        nixtla_times.append(time.time() - start_time)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        nixtla_peak_mem.append(peak / 2**20)

    results = pd.DataFrame(
        {
            "Data": datasets,
            "fastcan_time": fastcan_times,
            "fastcan_mem_mb": fastcan_peak_mem,
            "pysindy_time": pysindy_times,
            "pysindy_mem_mb": pysindy_peak_mem,
            "nixtla_time": nixtla_times,
            "nixtla_mem_mb": nixtla_peak_mem,
        }
    )

    results.to_csv("benchmarks_worst_case.csv", index=False)
    return results
