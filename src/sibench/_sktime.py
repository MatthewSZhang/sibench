from sktime.forecasting.auto_reg import AutoREG  
from sktime.forecasting.base import ForecastingHorizon
from sklearn.model_selection import TimeSeriesSplit
from rich.progress import Progress, TimeRemainingColumn
from nonlinear_benchmarks.error_metrics import RMSE, NRMSE, R_squared, MAE, fit_index
import numpy as np
import nonlinear_benchmarks
import pandas as pd
import optuna
import click
import sqlite3
import os


@click.command()
@click.option("--data", type=str, required=True, help="Name of the dataset")
@click.option("--n-folds", default=5, type=int, help="Number of folds for cross validation")
@click.option("--lags", default=2, type=int, help="Number of lags")
@click.option("--trend", default='n', type=click.Choice(['n', 'c', 't', 'ct']), help="Trend parameter")
def evaluate(
    data: str,
    n_folds: int,
    lags: int,
    trend: str,
):
    df_full, n_init = _get_data(data)

    r2 = _cross_validation(
        df_full,
        n_folds,
        n_init,
        lags,
        trend,
        print_results=True
    )
    print(f"R2: {r2:.4f}")
    return r2

@click.command()
@click.option("--data", type=str, required=True, help="Name of the dataset")
@click.option("--n-folds", default=5, type=int, help="Number of folds for cross validation")
@click.option("--max-lags", default=30, type=int, help="Maximum lags for AutoREG")
@click.option("--n-trials", default=None, type=int, help="Number of optimization trials")
def hpopt(
    data: str,
    n_folds: int,
    max_lags: int,
    n_trials: int,
):
    os.makedirs("results", exist_ok=True)
    db_path = os.path.abspath(f"results/sktime_{data}.db")

    df_full, n_init = _get_data(data)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")

    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///{db_path}",
        study_name="sktime_auto_reg",
        load_if_exists=True
    )
    
    print("Starting optimization...")
    study.optimize(lambda trial: _objective(
        trial,
        df_full,
        n_init,
        n_folds,
        max_lags,
    ), n_trials=n_trials)

    print("Best hyperparameters:")
    print(study.best_params)
    print(f"Best R2: {study.best_value:.4f}")

def _objective(
    trial,
    df_full,
    n_init,
    n_folds,
    max_lags,
):
    lags = trial.suggest_int('lags', 1, max_lags)
    trend = trial.suggest_categorical('trend', ['n', 'c', 't', 'ct'])


    r2 = _cross_validation(
        df_full,
        n_folds,
        n_init,
        lags,
        trend,
        print_results=False
    )
    return r2

def _get_data(data: str, return_test: bool = False, return_panel: bool = False):
    match data:
        case "EMPS":
            train_val, test = nonlinear_benchmarks.EMPS()
        case "CED":
            train_val, test = nonlinear_benchmarks.CED()
        case "WienerHammerBenchMark":
            train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
        case "Silverbox":
            train_val, test = nonlinear_benchmarks.Silverbox()
        case "F16":
            train_val, test = nonlinear_benchmarks.F16()
        case "ParWH":
            train_val, test = nonlinear_benchmarks.ParWH()
        case "Cascaded_Tanks":
            train_val, test = nonlinear_benchmarks.Cascaded_Tanks()
        case _:
            raise ValueError(f"Unknown dataset: {data}")

    if return_test:
        df_full = _prepare_data(test)
    else:
        df_full = _prepare_data(train_val)

    if isinstance(test, tuple):
        n_init = test[0].state_initialization_window_length
    else:
        n_init = test.state_initialization_window_length
    
    if return_panel:
        df_panel = []
        for i, df_data in enumerate(df_full):
            df_data["instances"] = i
            df_data["timepoints"] = np.arange(len(df_data))
            df_panel.append(df_data)
        df_full = pd.concat(df_panel)
        df_full = df_full.set_index(["instances", "timepoints"])
    
    return df_full, n_init

def _prepare_data(train_val):
    if not isinstance(train_val, tuple):
        train_val = (train_val,)
    df_full = [pd.DataFrame({"u": session.u, "y": session.y}) for session in train_val]

    return df_full


def test_opt(data, results_path):
    study = optuna.load_study(
        study_name="sktime_auto_reg",
        storage=f"sqlite:///{results_path}/sktime_{data}.db"
    )
    best_params = study.best_params

    score = test(
        data,
        lags=best_params['lags'],
        trend=best_params['trend'],
        results_path=results_path,
        return_metric="RMSE"
    )
    print(f"Test RMSE with best hyperparameters: {score:.4f}")
    return score

def test(data, lags, trend, results_path, return_metric: str = "RMSE"):
    df_train, _ = _get_data(data, return_panel=True)
    df_test, n_init = _get_data(data, return_test=True, return_panel=True)


    mdl = AutoREG(lags=lags, trend=trend)
    mdl.fit(y=df_train[["y"]], X=df_train[["u"]])
    y_hat = _predict(mdl, df_test)
    
    df_test["y_hat"] = y_hat
    y_true_list = []
    y_hat_list = []

    for _, group in df_test.groupby(level=0, sort=False):
        y_true_list.append(group["y"].values)
        y_hat_list.append(group["y_hat"].values)

    score = _compute_metrics(y_true_list, y_hat_list, n_init, print_results=True, return_metric=return_metric)
    return score

    

def _predict(mdl, df_data):
    if isinstance(df_data.index, pd.MultiIndex):
        y_hat_list = []
        for i, group in df_data.groupby(level=0, sort=False):
            fh = ForecastingHorizon(group.index.get_level_values(1), is_relative=False)
            y_pred = mdl.predict(X=group[["u"]], fh=fh).values

            divergence_indices = np.where((y_pred > 1e20) | np.isnan(y_pred) | np.isinf(y_pred))[0]
            if divergence_indices.size > 0:
                y_pred[divergence_indices[0]:] = 1e20
            y_hat_list.append(y_pred)
            
        y_hat = np.concatenate(y_hat_list)
        return y_hat
    else:
        fh = ForecastingHorizon(df_data.index, is_relative=False)
        y_hat = mdl.predict(X=df_data[["u"]], fh=fh).values

        divergence_indices = np.where((y_hat > 1e20) | np.isnan(y_hat) | np.isinf(y_hat))[0]
        if divergence_indices.size > 0:
            y_hat[divergence_indices[0]:] = 1e20

        return y_hat

def _cross_validation(df_full, n_folds, n_init, lags, trend, print_results=True):
    tscv = TimeSeriesSplit(n_splits=n_folds)
    y_hat_full = []
    y_true_full = []
        
    columns = [*Progress.get_default_columns()]
    columns[-1] = TimeRemainingColumn(elapsed_when_finished=True)
    with Progress(*columns, auto_refresh=True) as progress:
        series_task = progress.add_task("[green]Processing series...", total=len(df_full))

        for i, df_data in enumerate(df_full):
            fold_task = progress.add_task(f"[cyan]Series {i+1}/{len(df_full)}", total=n_folds)
            for train_index, val_index in tscv.split(df_data):
                df_train = df_data.iloc[train_index]
                df_val = df_data.iloc[val_index]

                mdl = AutoREG(lags=lags, trend=trend)

                mdl.fit(y=df_train["y"], X=df_train[["u"]])
                y_hat = _predict(mdl, df_val)

                y_true = df_val["y"].values
                y_hat_full.append(y_hat)
                y_true_full.append(y_true)
                progress.advance(fold_task)
            
            progress.remove_task(fold_task)
            progress.advance(series_task)
    return _compute_metrics(y_true_full, y_hat_full, n_init, print_results=print_results)


def _compute_metrics(y_true_full, y_pred_full, n_init, print_results = True, return_metric: str = "R-squared"):
    rmse = []
    nrmse = []
    r2 = []
    mae = []
    fidx = []
    n_sessions = len(y_pred_full)
    for i in range(n_sessions):
        n_steps = y_pred_full[i].shape[0]
        y_true = y_true_full[i][:n_steps]
        y_pred = y_pred_full[i]
        rmse.append(RMSE(y_true[n_init:], y_pred[n_init:]))
        nrmse.append(NRMSE(y_true[n_init:], y_pred[n_init:]))
        r2.append(R_squared(y_true[n_init:], y_pred[n_init:]))
        mae.append(MAE(y_true[n_init:], y_pred[n_init:]))
        fidx.append(fit_index(y_true[n_init:], y_pred[n_init:]))
    if print_results:
        print(f"RMSE: {np.mean(rmse)}")
        print(f"NRMSE: {np.mean(nrmse)}")
        print(f"R-squared: {np.mean(r2)}")
        print(f'MAE: {np.mean(mae)}')
        print(f"fit index: {np.mean(fidx)}")
    match return_metric:
        case "RMSE":
            return np.mean(rmse)
        case "NRMSE":
            return np.mean(nrmse)
        case "R-squared":
            return np.mean(r2)
        case "MAE":
            return np.mean(mae)
        case "fit_index":
            return np.mean(fidx)
        case _:
            raise ValueError(f"Unknown metric: {return_metric}")