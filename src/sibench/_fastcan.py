import nonlinear_benchmarks
from nonlinear_benchmarks.error_metrics import RMSE, NRMSE, R_squared, MAE, fit_index
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from fastcan.narx import NARX, make_narx
import optuna
import click

@click.command()
@click.option("--data", type=str, required=True, help="Name of the dataset")
@click.option("--n-folds", default=5, type=int, help="Number of folds")
@click.option("--n-terms", type=int, required=True, help="Number of terms")
@click.option("--max-delay", type=int, required=True, help="Maximum delay (n_lags)")
@click.option("--poly-degree", type=int, required=True, help="Polynomial degree")
def evaluate(
    data: str,
    n_folds,
    n_terms,
    max_delay,
    poly_degree,
):
    X_full, y_full, session_sizes_full, n_init = _get_train(data)

    scores = _cross_validation(
        X_full,
        y_full,
        n_init,
        session_sizes_full,
        n_folds,
        n_terms,
        max_delay,
        poly_degree,
        print_results=True,
    )
    avg_score = np.mean(scores)
    print(f"Average R2 over {n_folds} folds: {avg_score:.4f}")
    return avg_score

@click.command()
@click.option("--data", type=str, required=True, help="Name of the dataset")
@click.option("--n-folds", default=5, type=int, help="Number of folds for cross validation")
@click.option("--term-l", default=3, type=int, help="Lower bound for n_terms")
@click.option("--term-u", default=20, type=int, help="Upper bound for n_terms")
@click.option("--delay-l", default=3, type=int, help="Lower bound for max_delay")
@click.option("--delay-u", default=20, type=int, help="Upper bound for max_delay")
@click.option("--poly-l", default=1, type=int, help="Lower bound for poly_degree")
@click.option("--poly-u", default=3, type=int, help="Upper bound for poly_degree")
@click.option("--n-trials", default=None, type=int, help="Number of optimization trials")
def hpopt(
    data: str,
    n_folds: int,
    term_l: int,
    term_u: int,
    delay_l: int,
    delay_u: int,
    poly_l: int,
    poly_u: int,
    n_trials: int,
):
    X_full, y_full, session_sizes_full, n_init = _get_train(data)

    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///fastcan_{data}.db",
        study_name="fastcan_narx",
        load_if_exists=True
    )
    
    print("Starting optimization...")
    study.optimize(lambda trial: _objective(
        trial,
        X_full,
        y_full,
        n_init,
        session_sizes_full,
        n_folds=n_folds,
        term_l=term_l,
        term_u=term_u,
        delay_l=delay_l,
        delay_u=delay_u,
        poly_l=poly_l,
        poly_u=poly_u,
    ), n_trials=n_trials)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)


def _objective(
    trial,
    X_full,
    y_full,
    n_init,
    session_sizes_full,
    n_folds,
    term_l,
    term_u,
    delay_l,
    delay_u,
    poly_l,
    poly_u,
):
    n_terms = trial.suggest_int('n_terms', term_l, term_u)
    max_delay = trial.suggest_int('max_delay', delay_l, delay_u)
    poly_degree = trial.suggest_int('poly_degree', poly_l, poly_u)
    if max_delay > n_init:
        raise optuna.TrialPruned()

    r2 = []

    try:
        r2 = _cross_validation(
            X_full,
            y_full,
            n_init,
            session_sizes_full,
            n_folds=n_folds,
            n_terms=n_terms,
            n_lags=max_delay,
            n_polys=poly_degree,
            print_results=False,
        )
        return np.mean(r2)
    except Exception:
        return float('-inf')


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

    X_full, y_full, session_sizes_full = _prepare_data(train_val)

    if isinstance(test, tuple):
        n_init = test[0].state_initialization_window_length
    else:
        n_init = test.state_initialization_window_length
    return X_full, y_full, session_sizes_full, n_init

def _cross_validation(X_full, y_full, n_init, session_sizes_full, n_folds, n_terms, n_lags, n_polys, print_results=True):
    tscv = TimeSeriesSplit(n_splits=n_folds)
    scores = []
    for train_index, val_index in tscv.split(X_full):
        X_train, X_val = X_full[train_index], X_full[val_index]
        y_train, y_val = y_full[train_index], y_full[val_index]
        
        session_sizes_train = _split_sessions(train_index, session_sizes_full)
        session_sizes_val = _split_sessions(val_index, session_sizes_full)
        
        mdl_cv = make_narx(
            X = X_train,
            y = y_train,
            n_terms_to_select = n_terms,
            max_delay = n_lags,
            poly_degree = n_polys,
            session_sizes = session_sizes_train,
            max_candidates = 10000,
        )
        mdl_cv.fit(
            X=X_train,
            y=y_train,
            coef_init="one_step_ahead",
            session_sizes = session_sizes_train,
            verbose=2,
        )
        
        y_val_pred = _predict(
            mdl_cv,
            X_val,
            y_val,
            n_init,
            session_sizes_val,
        )
        
        scores.append(_compute_metrics(y_val, y_val_pred, n_init, session_sizes_val, print_results=print_results))
    return scores

def _prepare_data(train_val):
    if not isinstance(train_val, tuple):
        train_val = (train_val,)
    
    X_full = np.concatenate([session.u for session in train_val]).reshape(-1, 1)
    y_full = np.concatenate([session.y for session in train_val])
    session_sizes_full = [len(session.u) for session in train_val]
    return X_full, y_full, session_sizes_full

def _split_sessions(indices, full_sizes):
    if len(indices) == 0:
        return []
    start_idx = indices[0]
    end_idx = indices[-1] + 1
    
    session_sizes = []
    acc = 0
    for sz in full_sizes:
        inter_a = max(acc, start_idx)
        inter_b = min(acc+sz, end_idx)
        
        if inter_b > inter_a:
            session_sizes.append(inter_b - inter_a)
            
        acc += sz
        if acc >= end_idx:
            break
    return session_sizes

def _predict(mdl: NARX, X_full, y_full, n_init, session_sizes):
    if X_full.ndim == 1:
        X_full = X_full.reshape(-1, 1)
    max_delay = mdl.max_delay_
    if n_init < max_delay:
        raise ValueError(f"y_init length {n_init} is less than model max_delay {max_delay}")
    start_pos = n_init - max_delay
    y_hat_full = []
    session_cumsum = np.cumsum(session_sizes)
    for i, session in enumerate(session_cumsum):
        if i == 0:
            start_idx = 0
        else:
            start_idx = session_cumsum[i - 1]
        end_idx = session
        X = X_full[start_idx:end_idx]
        y_init = y_full[start_idx:start_idx + n_init]
        y_hat = mdl.predict(
            X = X[start_pos:],
            y_init = y_init[start_pos:],
        )
        y_hat = np.concatenate([y_init[:start_pos], y_hat])
        y_hat_full.append(y_hat)
    y_hat_full = np.concatenate(y_hat_full)
    return y_hat_full

def _compute_metrics(y_true_full, y_pred_full, n_init, session_sizes, print_results = True):
    rmse = []
    nrmse = []
    r2 = []
    mae = []
    fidx = []
    session_cumsum = np.cumsum(session_sizes)
    for i, session in enumerate(session_cumsum):
        if i == 0:
            start_idx = 0
        else:
            start_idx = session_cumsum[i - 1]
        end_idx = session
        y_true = y_true_full[start_idx:end_idx]
        y_pred = y_pred_full[start_idx:end_idx]
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