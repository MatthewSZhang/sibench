import nonlinear_benchmarks
from nonlinear_benchmarks.error_metrics import RMSE, NRMSE, R_squared, MAE, fit_index
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

import optuna
import click
from rich.progress import track


@click.command()
@click.option("--data", type=str, required=True, help="Data Name")
@click.option("--n_folds", type=int, default=5, help="Number of folds (ignored)")
@click.option("--season_length", type=int, default=50, help="Seasonality")
@click.option("--max_p", type=int, default=2, help="Max AR")
@click.option("--max_q", type=int, default=2, help="Max MA")
@click.option("--max_P", "max_P", type=int, default=1, help="Max Seasonal AR")
@click.option("--max_Q", "max_Q", type=int, default=1, help="Max Seasonal MA")
def evaluate(
    data: str,
    n_folds: int,
    season_length,
    max_p,
    max_q,
    max_P,
    max_Q,
):
    df_full, n_init, freq_str = _get_train(data)

    sf = _make_nixtla(
        season_length,
        max_p,
        max_q,
        max_P,
        max_Q,
        freq_str
    )

    r2 = _cross_validation(
        df_full,
        n_folds,
        n_init,
        sf,
        print_results=True
    )
    print(f"R2: {r2:.4f}")
    return r2


@click.command()
@click.option("--data", type=str, required=True, help="Data Name")
@click.option("--n_folds", type=int, default=5, help="Number of folds (ignored)")
@click.option("--season_max", type=int, default=100, help="Max Seasonality")
@click.option("--max_p", type=int, default=5, help="Max AR")
@click.option("--max_q", type=int, default=5, help="Max MA")
@click.option("--max_P", "max_P", type=int, default=1, help="Max Seasonal AR")
@click.option("--max_Q", "max_Q", type=int, default=1, help="Max Seasonal MA")
@click.option("--n_trials", type=int, default=None, help="Number of trials")
def hpopt(
    data: str,
    n_folds: int,
    season_max: int,
    max_p: int,
    max_q: int,
    max_P: int,
    max_Q: int,
    n_trials: int,
):
    df_full, n_init, freq_str = _get_train(data)

    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///results/nixtla_{data}.db",
        study_name="nixtla_autoarima",
        load_if_exists=True
    )
    
    print("Starting optimization...")
    study.optimize(lambda trial: _objective(
        trial,
        df_full,
        n_init,
        freq_str,
        n_folds,
        season_max,
        max_p,
        max_q,
        max_P,
        max_Q,
    ), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  R2: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study

def _objective(
    trial,
    df_full,
    n_init,
    freq_str,
    n_folds,
    season_max,
    max_p,
    max_q,
    max_P,
    max_Q,
):
    season_length = trial.suggest_int('season_length', 1, season_max)
    max_p_suggest = trial.suggest_int('max_p', 1, max_p)
    max_q_suggest = trial.suggest_int('max_q', 1, max_q)
    max_P_suggest = trial.suggest_int('max_P', 1, max_P)
    max_Q_suggest = trial.suggest_int('max_Q', 1, max_Q)

    sf = _make_nixtla(
        season_length,
        max_p_suggest,
        max_q_suggest,
        max_P_suggest,
        max_Q_suggest,
        freq_str
    )

    r2 = _cross_validation(
        df_full,
        n_folds,
        n_init,
        sf,
        print_results=False
    )
    return r2

def _get_train(data: str):
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

    if not isinstance(train_val, tuple):
        train_val = (train_val,)
    df_full = []
    for i in range(len(train_val)):
        freq_str = f"{int(float(train_val[i].sampling_time) * 1000000)}us"
        ds = pd.date_range(start='1970-01-01', periods=len(train_val[i].y), freq=freq_str)
        df_part = pd.DataFrame({
            "unique_id": f"{data}_{i}",  
            "ds": ds, 
            "y": train_val[i].y.flatten(), 
            "u": train_val[i].u.flatten()
        })
        df_full.append(df_part)

    if isinstance(test, tuple):
        n_init = test[0].state_initialization_window_length
    else:
        n_init = test.state_initialization_window_length
    return df_full, n_init, freq_str

def _make_nixtla(season_length, max_p, max_q, max_P, max_Q, freq_str):
    models = [AutoARIMA(
        season_length=season_length,  # 1 to 1000
        max_p=max_p, # 1 to 5 
        max_q=max_q, # 1 to 5
        max_P=max_P, # 1 to 5
        max_Q=max_Q, # 1 to 5
    )]
    sf = StatsForecast(
        models=models, 
        freq=freq_str, 
        n_jobs=1,
    )
    return sf

def _cross_validation(df_full, n_folds, n_init, sf, print_results=True):
    tscv = TimeSeriesSplit(n_splits=n_folds)
    y_hat_full = []
    y_true_full = []
        
    for i, df_data in enumerate(df_full):
        for train_index, val_index in track(
            tscv.split(df_data),
            total=n_folds,
            description=f"Processing time series {i+1}/{n_folds}"
        ):
            df_train = df_data.iloc[train_index]
            df_val = df_data.iloc[val_index]
            df_val_X = df_val.drop(columns=['y'])
            fcst = sf.forecast(df=df_train, h=len(df_val_X), X_df=df_val_X)
            y_hat = fcst["AutoARIMA"].values
            y_true = df_val["y"].values
            y_hat_full.append(y_hat)
            y_true_full.append(y_true)
    return _compute_metrics(y_true_full, y_hat_full, n_init, print_results=print_results)


def _compute_metrics(y_true_full, y_pred_full, n_init, print_results = True):
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
    return np.mean(r2)