import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# Keep imports lazy for yfinance in case network is unavailable
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

import gamspy as gp  # reuse the same modeling stack as the notebook
import gamspy.math as gpm


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_prices_and_returns(tickers, years=3, use_cached=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load prices/returns for the given tickers. If cached CSVs exist and use_cached is True,
    read them; otherwise fetch from yfinance.
    """
    prices_path = DATA_DIR / "prices.csv"
    returns_path = DATA_DIR / "returns.csv"

    if use_cached and prices_path.exists() and returns_path.exists():
        px = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        ret = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        # Align to requested tickers if subset/superset
        missing = [t for t in tickers if t not in ret.columns]
        if not missing:
            ret = ret[tickers]
            return px[tickers], ret

    if yf is None:
        raise RuntimeError("yfinance not available and cached data missing; set use_cached=True with existing CSVs.")

    px = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="any")
    ret = np.log(px / px.shift(1)).dropna()

    px.to_csv(prices_path)
    ret.to_csv(returns_path)
    return px, ret


def load_sectors(tickers) -> pd.Series:
    """Load sectors.csv and align to tickers."""
    sectors_path = DATA_DIR / "sectors.csv"
    if not sectors_path.exists():
        # Create a placeholder with 'Unknown' sectors
        return pd.Series({t: "Unknown" for t in tickers})
    df = pd.read_csv(sectors_path)
    if "ticker" in df.columns and "sector" in df.columns:
        ser = df.set_index("ticker")["sector"]
    else:
        # Assume two columns without headers
        df.columns = ["ticker", "sector"]
        ser = df.set_index("ticker")["sector"]
    ser = ser.reindex(tickers)
    ser = ser.fillna("Unknown")
    return ser


def solve_baseline_mean_cvar(ret_df: pd.DataFrame, sectors: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Baseline loss-based mean–CVaR model (no rebalance).
    """
    S = list(ret_df.columns)
    N = ret_df.shape[0]
    mu = ret_df.mean()
    alpha = params["alpha"]
    lambda_cvar = params["lambda_cvar"]
    lambda_mean = 1 - lambda_cvar
    if params["return_target"] == "equal_weight_mean":
        target_return = mu.mean()
    elif isinstance(params["return_target"], (float, int)):
        target_return = float(params["return_target"])
    elif params["return_target"] == "none":
        target_return = 0.0
    else:
        raise ValueError("Unsupported return_target")

    m = gp.Container(options=gp.Options(equation_listing_limit=10, relative_optimality_gap=0.01))
    i = gp.Set(m, name="i", records=S)
    s = gp.Set(m, name="s", records=range(N))

    R = gp.Parameter(m, name="R", domain=[s, i], records=ret_df.values)
    mu_param = gp.Parameter(m, name="mu", domain=[i], records=mu.values)

    w = gp.Variable(m, name="w", domain=[i], type="Positive")
    nu = gp.Variable(m, name="nu", type="Free")
    u = gp.Variable(m, name="u", domain=[s], type="Positive")

    budget = gp.Equation(m, name="budget"); budget[:] = gp.Sum(i, w[i]) == 1

    ret_con = gp.Equation(m, name="ret_con")
    ret_con[:] = gp.Sum(i, mu_param[i] * w[i]) >= target_return

    cvar_cons = gp.Equation(m, name="cvar_cons", domain=[s])
    cvar_cons[s] = u[s] >= -gp.Sum(i, R[s, i] * w[i]) - nu  # loss-based

    name_cap = gp.Equation(m, name="name_cap", domain=[i]); name_cap[i] = w[i] <= params["name_cap"]

    sector_cap_eqs = []
    if sectors is not None:
        for sector_name in sectors.unique():
            tickers_in_sector = [t for t in S if sectors.get(t) == sector_name]
            if not tickers_in_sector:
                continue
            safe = re.sub(r"[^A-Za-z0-9_]", "_", sector_name)
            mask_df = pd.DataFrame({"i": S, f"mask_{safe}": [1 if t in tickers_in_sector else 0 for t in S]})
            mask_param = gp.Parameter(m, name=f"mask_{safe}", domain=[i]); mask_param.setRecords(mask_df)
            cap = gp.Equation(m, name=f"cap_{safe}")
            cap[:] = gp.Sum(i, mask_param[i] * w[i]) <= params["sector_cap"]
            sector_cap_eqs.append(cap)

    cvar_term = nu + (1 / ((1 - alpha) * N)) * gp.Sum(s, u[s])
    mean_loss = -gp.Sum(i, mu_param[i] * w[i])
    obj = lambda_mean * mean_loss + lambda_cvar * cvar_term

    model = gp.Model(m, "baseline_mean_cvar",
                     equations= m.getEquations(),
                     problem=gp.Problem.LP,
                     sense=gp.Sense.MIN,
                     objective=obj)
    model.solve()

    w_opt = w.records.set_index("i")["level"]
    nu_val = float(nu.records["level"].iloc[0])
    cvar_val = float(cvar_term.toValue())
    mean_ret_val = float((mu * w_opt).sum())
    mean_loss_val = -mean_ret_val
    obj_val = float(obj.toValue())

    binding_names = w_opt[w_opt >= params["name_cap"] - 1e-6].index.tolist()
    binding_sectors = []
    for eq in sector_cap_eqs:
        rec = eq.records
        if rec.empty:
            continue
        upper = rec["upper"].replace([np.inf, -np.inf], np.nan)
        slack = upper - rec["level"]
        if abs(slack.iloc[0]) < 1e-6:
            binding_sectors.append(eq.name)

    return {
        "weights": w_opt,
        "mean_return": mean_ret_val,
        "mean_loss": mean_loss_val,
        "VaR": nu_val,
        "target_return": target_return,
        "CVaR_loss": cvar_val,
        "objective": obj_val,
        "binding_names": binding_names,
        "binding_sectors": binding_sectors,
    }


def solve_rebalance_mean_cvar(ret_df: pd.DataFrame, sectors: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rebalance model: loss-based mean–CVaR with one planned rebalance, costs in the objective.
    """
    S = list(ret_df.columns)
    N = ret_df.shape[0]
    alpha = params["alpha"]
    lambda_cvar = params["lambda_cvar"]
    lambda_mean = 1 - lambda_cvar
    if params["return_target"] == "equal_weight_mean":
        target_return = ret_df.mean().mean()
    elif isinstance(params["return_target"], (float, int)):
        target_return = float(params["return_target"])
    elif params["return_target"] == "none":
        target_return = 0.0
    else:
        raise ValueError("Unsupported return_target")
    turnover_cap = params.get("turnover_cap", 1.0)

    # rebalance index
    rb = params["rebalance_day"]
    if rb == "midpoint":
        k = N // 2
    elif isinstance(rb, int):
        k = max(1, min(rb, N - 1))
    else:
        raise ValueError("rebalance_day not understood")

    ret_pre_full = ret_df.copy();  ret_pre_full.iloc[k:, :] = 0.0
    ret_post_full = ret_df.copy(); ret_post_full.iloc[:k, :] = 0.0

    m2 = gp.Container(options=gp.Options(equation_listing_limit=10, relative_optimality_gap=0.01))
    i = gp.Set(m2, name="i", records=S)
    s = gp.Set(m2, name="s", records=range(N))

    Rpre = gp.Parameter(m2, name="Rpre", domain=[s, i], records=ret_pre_full.values)
    Rpost = gp.Parameter(m2, name="Rpost", domain=[s, i], records=ret_post_full.values)
    mu_pre = gp.Parameter(m2, name="mu_pre", domain=[i], records=ret_df.iloc[:k, :].mean().values)
    mu_post = gp.Parameter(m2, name="mu_post", domain=[i], records=ret_df.iloc[k:, :].mean().values)

    w0   = gp.Variable(m2, name="w0", domain=[i], type="Positive")
    w1   = gp.Variable(m2, name="w1", domain=[i], type="Positive")
    buy  = gp.Variable(m2, name="buy", domain=[i], type="Positive")
    sell = gp.Variable(m2, name="sell", domain=[i], type="Positive")
    t    = gp.Variable(m2, name="t", domain=[i], type="Positive")
    nu   = gp.Variable(m2, name="nu", type="Free")
    u    = gp.Variable(m2, name="u", domain=[s], type="Positive")

    budget0 = gp.Equation(m2, name="budget0"); budget0[:] = gp.Sum(i, w0[i]) == 1
    budget1 = gp.Equation(m2, name="budget1"); budget1[:] = gp.Sum(i, w1[i]) == 1  # costs in objective

    rebalance = gp.Equation(m2, name="rebalance", domain=[i]); rebalance[i] = w1[i] == w0[i] + buy[i] - sell[i]
    trade_pos = gp.Equation(m2, name="trade_pos", domain=[i]); trade_pos[i] = t[i] >= w1[i] - w0[i]
    trade_neg = gp.Equation(m2, name="trade_neg", domain=[i]); trade_neg[i] = t[i] >= w0[i] - w1[i]
    t_buy     = gp.Equation(m2, name="t_buy",  domain=[i]); t_buy[i]  = t[i] >= buy[i]
    t_sell    = gp.Equation(m2, name="t_sell", domain=[i]); t_sell[i] = t[i] >= sell[i]
    t_total_cap = gp.Equation(m2, name="t_total_cap"); t_total_cap[:] = gp.Sum(i, t[i]) <= turnover_cap

    ret_con = gp.Equation(m2, name="ret_con")
    ret_con[:] = (k / N) * gp.Sum(i, mu_pre[i] * w0[i]) + ((N - k) / N) * gp.Sum(i, mu_post[i] * w1[i]) >= target_return

    cvar_cons = gp.Equation(m2, name="cvar_cons", domain=[s])
    cvar_cons[s] = u[s] >= -gp.Sum(i, Rpre[s, i] * w0[i] + Rpost[s, i] * w1[i]) - nu

    name_cap0 = gp.Equation(m2, name="name_cap0", domain=[i]); name_cap0[i] = w0[i] <= params["name_cap"]
    name_cap1 = gp.Equation(m2, name="name_cap1", domain=[i]); name_cap1[i] = w1[i] <= params["name_cap"]

    sector_cap_eqs2 = []
    if sectors is not None:
        for sector_name in sectors.unique():
            tickers_in_sector = [t for t in S if sectors.get(t) == sector_name]
            if not tickers_in_sector:
                continue
            safe = re.sub(r"[^A-Za-z0-9_]", "_", sector_name)
            mask_df = pd.DataFrame({"i": S, f"mask_{safe}": [1 if t in tickers_in_sector else 0 for t in S]})
            mask_param = gp.Parameter(m2, name=f"mask_{safe}", domain=[i]); mask_param.setRecords(mask_df)
            cap0 = gp.Equation(m2, name=f"cap0_{safe}"); cap0[:] = gp.Sum(i, mask_param[i] * w0[i]) <= params["sector_cap"]
            cap1 = gp.Equation(m2, name=f"cap1_{safe}"); cap1[:] = gp.Sum(i, mask_param[i] * w1[i]) <= params["sector_cap"]
            sector_cap_eqs2.extend([cap0, cap1])

    cvar_term = nu + (1 / ((1 - alpha) * N)) * gp.Sum(s, u[s])
    mean_loss = (k / N) * (-gp.Sum(i, mu_pre[i] * w0[i])) + ((N - k) / N) * (-gp.Sum(i, mu_post[i] * w1[i]))
    cost_term = (params["buy_cost"] + params["sell_cost"]) * gp.Sum(i, t[i])
    obj = lambda_mean * mean_loss + lambda_cvar * cvar_term + cost_term

    model2 = gp.Model(m2, "rebalance_mean_cvar",
                      equations= m2.getEquations(),
                      problem=gp.Problem.LP,
                      sense=gp.Sense.MIN,
                      objective=obj)
    model2.solve()

    w0_opt = w0.records.set_index("i")["level"]
    w1_opt = w1.records.set_index("i")["level"]
    buy_opt = buy.records.set_index("i")["level"]
    sell_opt = sell.records.set_index("i")["level"]
    t_opt = t.records.set_index("i")["level"]
    nu_val = float(nu.records["level"].iloc[0])
    cvar_val = float(cvar_term.toValue())
    mean_loss_val = float(mean_loss.toValue())
    mean_ret_val = -mean_loss_val
    obj_val = float(obj.toValue())
    turnover = float(t_opt.sum())

    binding_names0 = w0_opt[w0_opt >= params["name_cap"] - 1e-6].index.tolist()
    binding_names1 = w1_opt[w1_opt >= params["name_cap"] - 1e-6].index.tolist()
    binding_sectors = []
    for eq in sector_cap_eqs2:
        rec = eq.records
        if rec.empty:
            continue
        upper = rec["upper"].replace([np.inf, -np.inf], np.nan)
        slack = upper - rec["level"]
        if abs(slack.iloc[0]) < 1e-6:
            binding_sectors.append(eq.name)

    return {
        "w0": w0_opt,
        "w1": w1_opt,
        "buy": buy_opt,
        "sell": sell_opt,
        "target_return": target_return,
        "turnover_by_name": t_opt,
        "turnover": turnover,
        "mean_return": mean_ret_val,
        "mean_loss": mean_loss_val,
        "VaR": nu_val,
        "CVaR_loss": cvar_val,
        "objective": obj_val,
        "binding_names0": binding_names0,
        "binding_names1": binding_names1,
        "binding_sectors": binding_sectors,
        "k": k,
    }
