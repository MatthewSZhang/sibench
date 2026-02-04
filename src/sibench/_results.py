import pandas as pd
import click
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
