import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


@click.command()
@click.option(
    "--opt-csv",
    required=True,
    type=click.Path(exists=True),
    help="Path to optimization results CSV (NRMSE)",
)
@click.option(
    "--time-mem-folder",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to folder containing time and memory results",
)
def plot_results(opt_csv, time_mem_folder):
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "legend.title_fontsize": 18,
        }
    )

    # 1. Process NRMSE Data
    print(f"Reading NRMSE results from {opt_csv}...")
    nrmse_df = pd.read_csv(opt_csv)

    # Melt for plotting
    nrmse_long = nrmse_df.melt(
        id_vars=["Data"],
        value_vars=["fastcan", "pysindy", "nixtla"],
        var_name="Model",
        value_name="NRMSE",
    )

    # Rename for plotting
    nrmse_long["Data"] = nrmse_long["Data"].replace(
        {"WienerHammerBenchMark": "WH", "Cascaded_Tanks": "CT"}
    )
    nrmse_long["Model"] = nrmse_long["Model"].replace(
        {"pysindy": "PySINDy", "nixtla": "Nixtla"}
    )

    # Plot NRMSE
    plt.figure(figsize=(10, 6))
    sns.barplot(data=nrmse_long, x="Data", y="NRMSE", hue="Model", palette="viridis")
    plt.yscale("log")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("fig_nrmse.pdf")
    plt.close()
    print("Saved fig_nrmse.pdf")

    # 2. Process Time & Memory Data
    print(f"Reading time/memory results from {time_mem_folder}...")
    models = ["fastcan", "pysindy", "nixtla"]

    performance_data = []

    for model in models:
        for dataset in datasets:
            file_path = os.path.join(time_mem_folder, f"{model}_{dataset}.csv")
            if os.path.exists(file_path):
                try:
                    df_temp = pd.read_csv(file_path)
                    # Expected columns: Data, {model}_time, {model}_mem_mb
                    time_col = f"{model}_time"
                    mem_col = f"{model}_mem_mb"

                    if time_col in df_temp.columns and mem_col in df_temp.columns:
                        val_time = df_temp[time_col].values[0]
                        val_mem = df_temp[mem_col].values[0]

                        performance_data.append(
                            {
                                "Data": dataset,
                                "Model": model,
                                "Time (s)": val_time,
                                "Memory (MB)": val_mem,
                            }
                        )
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            else:
                print(f"Warning: File not found {file_path}")

    perf_df = pd.DataFrame(performance_data)

    # Rename for plotting
    perf_df["Data"] = perf_df["Data"].replace(
        {"WienerHammerBenchMark": "WH", "Cascaded_Tanks": "CT"}
    )
    perf_df["Model"] = perf_df["Model"].replace(
        {"pysindy": "PySINDy", "nixtla": "Nixtla"}
    )

    if perf_df.empty:
        print("No time/memory data found!")
        return

    # Plot Time
    plt.figure(figsize=(10, 6))
    sns.barplot(data=perf_df, x="Data", y="Time (s)", hue="Model", palette="viridis")
    plt.yscale("log")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("fig_time.pdf")
    plt.close()
    print("Saved fig_time.pdf")

    # Plot Memory
    plt.figure(figsize=(10, 6))
    sns.barplot(data=perf_df, x="Data", y="Memory (MB)", hue="Model", palette="viridis")
    plt.yscale("log")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("fig_memory.pdf")
    plt.close()
    print("Saved fig_memory.pdf")
