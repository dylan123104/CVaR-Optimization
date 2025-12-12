import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# With server.py under Project/app, base_dir points to Project
BASE_DIR = Path(__file__).resolve().parent.parent
# Add Project root to sys.path so `src` is importable
sys.path.insert(0, str(BASE_DIR))
PUBLIC_DIR = BASE_DIR / "public"

from src.models import (
    load_prices_and_returns,
    load_sectors,
    solve_baseline_mean_cvar,
    solve_rebalance_mean_cvar,
)

app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path="/")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


def _parse_params(data: dict):
    # tickers can come as list or comma-separated string
    tickers = data.get("tickers", [])
    if isinstance(tickers, str):
        tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    use_cached = bool(data.get("use_cached", True))
    years = int(data.get("years", 3))
    lambda_cvar = float(data.get("lambda_cvar", 0.8))
    rb_raw = data.get("rebalance_day", "midpoint")
    if isinstance(rb_raw, str) and rb_raw.strip().isdigit():
        rb_val = int(rb_raw.strip())
    else:
        rb_val = "midpoint"
    params = {
        "alpha": float(data.get("alpha", 0.95)),
        "lambda_cvar": lambda_cvar,
        "lambda_mean": 1 - lambda_cvar,  # derived
        "name_cap": float(data.get("name_cap", 0.2)),
        "sector_cap": float(data.get("sector_cap", 0.5)),
        "buy_cost": float(data.get("buy_cost", 0.002)),
        "sell_cost": float(data.get("sell_cost", 0.002)),
        "turnover_cap": float(data.get("turnover_cap", 1.0)),
        "rebalance_day": rb_val,
        "return_target": data.get("return_target", "none"),
        "solver": data.get("solver", "highs"),
    }
    return tickers, years, use_cached, params


def _portfolio_losses(ret_df: pd.DataFrame, weights: pd.Series):
    w = weights.reindex(ret_df.columns).fillna(0).values
    return -(ret_df.values @ w)


def _portfolio_losses_rebalance(ret_df: pd.DataFrame, res_reb: dict):
    k = res_reb["k"]
    w0 = res_reb["w0"].reindex(ret_df.columns).fillna(0).values
    w1 = res_reb["w1"].reindex(ret_df.columns).fillna(0).values
    return np.concatenate([
        -(ret_df.iloc[:k].values @ w0),
        -(ret_df.iloc[k:].values @ w1),
    ])


def _cvar_stats(losses: np.ndarray, alpha: float):
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var
    return {
        "mean_loss": float(losses.mean()),
        "VaR": var,
        "CVaR": cvar,
        "max_loss": float(losses.max()),
        "losses": losses.tolist(),
    }


@app.route("/api/run", methods=["POST"])
def run_models():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    tickers, years, use_cached, model_params = _parse_params(data)
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    try:
        _, ret = load_prices_and_returns(tickers, years=years, use_cached=use_cached)
    except Exception as e:
        return jsonify({"error": f"Data load failed: {e}"}), 500

    sectors = load_sectors(tickers)

    # Baseline
    try:
        res_base = solve_baseline_mean_cvar(ret, sectors, model_params)
    except Exception as e:
        return jsonify({"error": f"Baseline solve failed: {e}"}), 500

    base_losses = _portfolio_losses(ret, res_base["weights"])
    base_stats = _cvar_stats(base_losses, model_params["alpha"])

    # Rebalance
    try:
        res_reb = solve_rebalance_mean_cvar(ret, sectors, model_params)
    except Exception as e:
        return jsonify({"error": f"Rebalance solve failed: {e}"}), 500

    reb_losses = _portfolio_losses_rebalance(ret, res_reb)
    reb_stats = _cvar_stats(reb_losses, model_params["alpha"])

    def series_to_dict(ser):
        return {k: float(v) for k, v in ser.items()}

    response = {
        "baseline": {
        "mean_return": res_base["mean_return"],
        "mean_loss": res_base["mean_loss"],
        "VaR": res_base["VaR"],
        "CVaR_loss": res_base["CVaR_loss"],
        "objective": res_base["objective"],
        "target_return": res_base.get("target_return", 0.0),
        "binding_names": res_base.get("binding_names", []),
        "binding_sectors": res_base.get("binding_sectors", []),
        "weights": series_to_dict(res_base["weights"]),
        "sector_totals": series_to_dict(
            res_base["weights"].to_frame("w")
            .join(sectors.rename("sector"))
            .groupby("sector")["w"].sum()
            ),
            "name_cap": model_params["name_cap"],
            "sector_cap": model_params["sector_cap"],
            "losses": base_stats["losses"],
            "VaR_plot": base_stats["VaR"],
            "CVaR_plot": base_stats["CVaR"],
            "mean_loss_plot": base_stats["mean_loss"],
            "max_loss": base_stats["max_loss"],
        },
        "rebalance": {
            "mean_return": res_reb["mean_return"],
            "mean_loss": res_reb["mean_loss"],
            "VaR": res_reb["VaR"],
        "CVaR_loss": res_reb["CVaR_loss"],
        "objective": res_reb["objective"],
        "turnover": res_reb["turnover"],
        "k": res_reb["k"],
        "target_return": res_reb.get("target_return", 0.0),
        "binding_names0": res_reb.get("binding_names0", []),
        "binding_names1": res_reb.get("binding_names1", []),
        "binding_sectors": res_reb.get("binding_sectors", []),
        "w0": series_to_dict(res_reb["w0"]),
        "w1": series_to_dict(res_reb["w1"]),
        "buy": series_to_dict(res_reb["buy"]),
        "sell": series_to_dict(res_reb["sell"]),
        "sector_totals": series_to_dict(
                res_reb["w1"].to_frame("w1")
                .join(sectors.rename("sector"))
                .groupby("sector")["w1"].sum()
            ),
            "name_cap": model_params["name_cap"],
            "sector_cap": model_params["sector_cap"],
            "turnover_cap": model_params.get("turnover_cap", 1.0),
            "losses": reb_stats["losses"],
            "VaR_plot": reb_stats["VaR"],
            "CVaR_plot": reb_stats["CVaR"],
            "mean_loss_plot": reb_stats["mean_loss"],
            "max_loss": reb_stats["max_loss"],
        },
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
