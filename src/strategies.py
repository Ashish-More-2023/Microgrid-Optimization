"""
Step 6 - Control strategies compared in this project.

S0 No battery      : renewables + grid only (reference)
S1 Rule-based      : greedy self-consumption (charge on surplus, discharge on deficit)
S2 LP cost-only    : perfect-foresight LP minimising energy cost only
S3 LP multi-obj    : perfect-foresight LP with cost + battery wear + peak demand
S4 MPC multi-obj   : rolling 24 h horizon, re-optimised every 15 min using
                     causal forecasts (realistic, implementable controller)
"""
import time
import numpy as np
import pandas as pd

from .config import ProjectConfig, DT_H
from .optimizer import solve_dispatch
from .forecasting import causal_climatology
from .battery import assess_degradation


def _settle_grid(net_after_batt, g):
    """Given residual demand (+) or surplus (-) after the battery, settle with the grid."""
    imp = np.clip(net_after_batt, 0, None)
    surplus = np.clip(-net_after_batt, 0, None)
    exp = np.minimum(surplus, g.export_limit_kw)
    cur = surplus - exp
    return imp, exp, cur


def no_battery(sc: pd.DataFrame, cfg: ProjectConfig):
    net = (sc.load_kw - sc.ren_kw).to_numpy()
    imp, exp, cur = _settle_grid(net, cfg.grid)
    T = len(sc)
    e = np.full(T, cfg.battery.e_init)
    return dict(p_imp=imp, p_exp=exp, p_ch=np.zeros(T), p_dis=np.zeros(T), p_cur=cur, energy=e)


def rule_based(sc: pd.DataFrame, cfg: ProjectConfig):
    b = cfg.battery
    net = (sc.load_kw - sc.ren_kw).to_numpy()
    T = len(sc)
    ch, dis, e = np.zeros(T), np.zeros(T), np.zeros(T)
    soc = b.e_init
    for t in range(T):
        if net[t] < 0:   # surplus -> charge
            room = (b.e_max - soc) / (b.eta_ch * DT_H)
            ch[t] = min(-net[t], b.p_max_kw, room)
        else:            # deficit -> discharge
            avail = (soc - b.e_min) * b.eta_dis / DT_H
            dis[t] = min(net[t], b.p_max_kw, avail)
        soc = soc + b.eta_ch * DT_H * ch[t] - DT_H / b.eta_dis * dis[t]
        e[t] = soc
    imp, exp, cur = _settle_grid(net + ch - dis, cfg.grid)
    return dict(p_imp=imp, p_exp=exp, p_ch=ch, p_dis=dis, p_cur=cur, energy=e)


def lp_perfect(sc: pd.DataFrame, cfg: ProjectConfig, deg_weight, use_demand_charge):
    return solve_dispatch(sc.load_kw, sc.ren_kw, sc.price_import, sc.price_export, sc.month,
                          cfg, deg_weight=deg_weight, use_demand_charge=use_demand_charge)


def mpc(sc: pd.DataFrame, cfg: ProjectConfig, price_known=True, verbose=True):
    """Receding-horizon MPC. At step t the current load/renewables are measured;
    future steps use causal climatology forecasts. Only the first decision is applied."""
    b, H = cfg.battery, cfg.optim.mpc_horizon_steps
    T = len(sc)
    load, ren = sc.load_kw.to_numpy(), sc.ren_kw.to_numpy()
    f_load = causal_climatology(sc.load_kw).to_numpy()
    f_ren = causal_climatology(sc.ren_kw).to_numpy()
    pi = sc.price_import.to_numpy()
    f_pi = pi if price_known else causal_climatology(sc.price_import).to_numpy()
    months = sc.month.to_numpy()
    pe_all = np.minimum(cfg.grid.feed_in_tariff, 0.9 * f_pi)

    ch, dis, e_hist = np.zeros(T), np.zeros(T), np.zeros(T)
    running_peak = {}
    soc = b.e_init
    t0 = time.time()
    for t in range(T):
        sl = slice(t, min(t + H, T))
        L = f_load[sl].copy(); R = f_ren[sl].copy(); P = f_pi[sl].copy(); PE = pe_all[sl].copy()
        L[0], R[0], P[0] = load[t], ren[t], pi[t]
        PE[0] = min(cfg.grid.feed_in_tariff, 0.9 * pi[t])
        m = months[sl]
        uniq = np.unique(m)
        floor = [running_peak.get(mm, 0.0) for mm in uniq]
        sol = solve_dispatch(L, R, P, PE, m, cfg, e0=soc, peak_floor=floor,
                             terminal=False)
        ch[t], dis[t] = sol["p_ch"][0], sol["p_dis"][0]
        soc = soc + b.eta_ch * DT_H * ch[t] - DT_H / b.eta_dis * dis[t]
        soc = min(max(soc, b.e_min), b.e_max)
        e_hist[t] = soc
        imp_t = max(load[t] - ren[t] + ch[t] - dis[t], 0.0)
        running_peak[months[t]] = max(running_peak.get(months[t], 0.0), imp_t)
        if verbose and t % 960 == 0:
            print(f"      MPC step {t}/{T}  ({time.time() - t0:.0f}s)")
    net = load - ren + ch - dis
    imp, exp, cur = _settle_grid(net, cfg.grid)
    # MPC does not end at the starting SOC; the KPI function credits/charges the
    # stored-energy difference so all strategies are compared fairly.
    return dict(p_imp=imp, p_exp=exp, p_ch=ch, p_dis=dis, p_cur=cur, energy=e_hist)


def compute_kpis(name, sc: pd.DataFrame, res: dict, cfg: ProjectConfig) -> dict:
    b, g = cfg.battery, cfg.grid
    imp, exp, ch, dis, cur, e = (np.asarray(res[k]) for k in ["p_imp", "p_exp", "p_ch", "p_dis", "p_cur", "energy"])
    pi, pe = sc.price_import.to_numpy(), sc.price_export.to_numpy()
    energy_cost = float(np.sum(DT_H * (pi * imp - pe * exp)))
    # value any change in stored energy at the mean import price (fair end-of-period comparison)
    stored_delta = float(e[-1] - b.e_init)
    energy_cost -= stored_delta * float(np.mean(pi))
    monthly_peak = pd.Series(imp, index=sc.index).groupby(sc.month.values).max()
    demand_cost = float(monthly_peak.sum() * g.demand_charge_inr_per_kw_month)
    deg = assess_degradation(e, ch, dis, b)
    throughput_cost = deg["throughput_kwh"] * b.deg_cost_per_kwh_throughput

    ren_total = float(np.sum(sc.ren_kw) * DT_H)
    load_total = float(np.sum(sc.load_kw) * DT_H)
    ren_used = ren_total - float(np.sum(exp + cur) * DT_H)
    return {
        "strategy": name,
        "energy_cost_inr": energy_cost,
        "demand_charge_inr": demand_cost,
        "battery_wear_cost_inr": throughput_cost,
        "total_cost_inr": energy_cost + demand_cost + throughput_cost,
        "grid_import_mwh": float(np.sum(imp) * DT_H / 1000),
        "grid_export_mwh": float(np.sum(exp) * DT_H / 1000),
        "curtailed_mwh": float(np.sum(cur) * DT_H / 1000),
        "self_consumption_pct": 100 * ren_used / ren_total if ren_total > 0 else 0.0,
        "self_sufficiency_pct": 100 * (load_total - float(np.sum(imp) * DT_H)) / load_total,
        "avg_monthly_peak_import_kw": float(monthly_peak.mean()),
        "co2_tonnes": float(np.sum(imp) * DT_H * g.emission_factor_kg_per_kwh / 1000),
        **deg,
    }
