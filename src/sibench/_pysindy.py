import nonlinear_benchmarks
from nonlinear_benchmarks.error_metrics import RMSE, NRMSE, R_squared, MAE, fit_index
from sklearn.model_selection import KFold
import numpy as np
import pysindy as ps
from pysindy.differentiation import FiniteDifference
from pysindy import SINDy
import optuna
import click
from rich.progress import track


@click.command()
@click.option("--data", type=str, required=True, help="Name of the dataset")
@click.option("--d-l", default=1, type=int, help="PolynomialLibrary: Lower bound for degree")
@click.option("--d-u", default=3, type=int, help="PolynomialLibrary: Upper bound for degree")
@click.option("--f-l", default=1, type=int, help="FourierLibrary: Lower bound for frequency")
@click.option("--f-u", default=10, type=int, help="FourierLibrary: Upper bound for frequency")
@click.option("--o-l", default=1, type=int, help="FiniteDifference: Lower bound for order")
@click.option("--o-u", default=3, type=int, help="FiniteDifference: Upper bound for order")
@click.option("--t-l", default=1e-6, type=float, help="STLSQ: Lower bound for threshold")
@click.option("--t-u", default=1.0, type=float, help="STLSQ: Upper bound for threshold")
@click.option("--a-l", default=1e-6, type=float, help="STLSQ: Lower bound for alpha")
@click.option("--a-u", default=1.0, type=float, help="STLSQ: Upper bound for alpha")
@click.option("--rtol", default=1e-3, type=float, help="Integrator relative tolerance")
@click.option("--atol", default=1e-3, type=float, help="Integrator absolute tolerance")
@click.option("--n-trials", default=None, type=int, help="Number of optimization trials")
def hpopt(
    data: str,
    d_l: int,
    d_u: int,
    f_l: int,
    f_u: int,
    o_l: int,
    o_u: int,
    t_l: float,
    t_u: float,
    a_l: float,
    a_u: float,
    rtol: float,
    atol: float,
    n_trials: int,
):
    integrator_kws = {"method": "LSODA", "rtol": rtol, "atol": atol}
    X_full, y_full, dt_full, n_init = _get_train(data)

    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///pysindy_{data}.db",
        study_name="pysindy_stlsq",
        load_if_exists=True
    )
    
    print("Starting optimization...")
    study.optimize(lambda trial: _objective(
        trial,
        X_full,
        y_full,
        dt_full,
        n_init,
        d_l,
        d_u,
        f_l,
        f_u,
        o_l,
        o_u,
        t_l,
        t_u,
        a_l,
        a_u,
        integrator_kws,
    ), n_trials=n_trials)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)


def _objective(
    trial,
    X_full,
    y_full,
    dt_full,
    n_init,
    d_l,
    d_u,
    f_l,
    f_u,
    o_l,
    o_u,
    t_l,
    t_u,
    a_l,
    a_u,
    integrator_kws,

):
    degree = trial.suggest_int('degree', d_l, d_u)
    freq = trial.suggest_int('freq', f_l, f_u)
    order = trial.suggest_int('order', o_l, o_u)
    threshold = trial.suggest_float('threshold', t_l, t_u, log=True)
    alpha = trial.suggest_float('alpha', a_l, a_u, log=True)

    r2 = []

    try:
        r2 = _cross_validation(
            X_full,
            y_full,
            dt_full,
            n_init,
            degree,
            freq,
            order,
            threshold,
            alpha,
            integrator_kws=integrator_kws,
            print_results=True,
        )
        return np.mean(r2)
    except Exception:
        return float('-inf')


@click.command()
@click.option("--data", type=str, required=True, help="Name of the dataset")
@click.option("--degree", type=int, required=True, help="Number of polynomial degrees")
@click.option("--freq", type=int, required=True, help="Number of Fourier frequencies")
@click.option("--order", type=int, required=True, help="Order of finite difference")
@click.option("-t", "--threshold", type=float, required=True, help="SINDy threshold")
@click.option("-a", "--alpha", type=float, required=True, help="SINDy alpha")
@click.option("--rtol", default=1e-3, type=float, help="Integrator relative tolerance")
@click.option("--atol", default=1e-3, type=float, help="Integrator absolute tolerance")
def evaluate(
    data: str,
    degree,
    freq,
    order,
    threshold,
    alpha,
    rtol,
    atol,
):
    X_full, y_full, dt_full, n_init = _get_train(data)

    integrator_kws = {'method': 'LSODA', 'rtol': rtol, 'atol': atol}

    scores = _cross_validation(
        X_full,
        y_full,
        dt_full,
        n_init,
        degree,
        freq,
        order,
        threshold,
        alpha,
        integrator_kws=integrator_kws,
        print_results=True,
    )
    avg_score = np.mean(scores)
    print(f"CV R2: {avg_score:.4f}")
    return avg_score

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

    X_full, y_full, dt_full = _prepare_data(train_val)

    if isinstance(test, tuple):
        n_init = test[0].state_initialization_window_length
    else:
        n_init = test.state_initialization_window_length
    return X_full, y_full, dt_full, n_init

def _prepare_data(train_val):
    if not isinstance(train_val, tuple):
        train_val = (train_val,)
    X_full = [session.u.reshape(-1, 1) for session in train_val]
    y_full = [session.y.reshape(-1, 1) for session in train_val]
    dt_full = [session.sampling_time for session in train_val]
    return X_full, y_full, dt_full


def _compute_metrics(y_true_full, y_pred_full, n_init, print_results = True):
    rmse = []
    nrmse = []
    r2 = []
    mae = []
    fidx = []
    n_sessions = len(y_pred_full)
    n_steps = y_pred_full[0].shape[0]
    for i in range(n_sessions):
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

def _make_sindyc(X_full, y_full, dt_full, n_degrees, n_freqs, n_orders, threshold, alpha):
    poly_library = ps.PolynomialLibrary(degree=n_degrees)
    fourier_library = ps.FourierLibrary(n_frequencies=n_freqs)
    combined_library = poly_library * fourier_library

    finite_difference = FiniteDifference(order=n_orders)

    stlsq_optimizer = ps.STLSQ(
        threshold = threshold,
        alpha=alpha,
    )
    model = ps.SINDy(
        optimizer=stlsq_optimizer,
        feature_library=combined_library,
        differentiation_method=finite_difference,
    )
    model.fit(y_full, t=dt_full, u=X_full)
    return model

def _predict(model: SINDy, X_full, y_full, dt_full, n_steps, integrator_kws):
    def u_func(t, u, dt):
        return u[np.round(t / dt).astype(int)]
    
    y_hat_full = []
    for i, dt in enumerate(dt_full):
        t_steps = np.arange(0, dt*n_steps, dt)
        y_hat = model.simulate(
            y_full[i][0], 
            t=t_steps, 
            u=lambda t: u_func(t, X_full[i], dt),
            integrator_kws=integrator_kws
        )
        y_hat_full.append(y_hat)
    return y_hat_full

def _cross_validation(
    X_full, y_full, dt_full, n_init, n_degrees, n_freqs, n_orders, threshold, alpha, integrator_kws, print_results=True
):
    n_steps = 100 # Too slow to simulate full length of validation data
    n_sessions = len(X_full)
    
    
    if n_sessions < 2:
        splits = [([0], [0])]
    else:
        kf = KFold(n_splits=min(n_sessions, 3))
        splits = list(kf.split(X_full))
        
    scores = []
    for train_index, val_index in track(splits, description="Processing folds"):
        X_train, X_val = [X_full[i] for i in train_index], [X_full[i] for i in val_index]
        y_train, y_val = [y_full[i] for i in train_index], [y_full[i] for i in val_index]
        dt_train, dt_val = [dt_full[i] for i in train_index], [dt_full[i] for i in val_index]

        
        mdl_cv = _make_sindyc(
            X_train,
            y_train,
            dt_train,
            n_degrees,
            n_freqs,
            n_orders,
            threshold,
            alpha,
        )
        
        y_val_pred = _predict(
            mdl_cv,
            X_val,
            y_val,
            dt_val,
            n_steps=n_steps,
            integrator_kws=integrator_kws,
        )
        
        scores.append(_compute_metrics(y_val, y_val_pred, n_init, print_results=print_results))
    return scores