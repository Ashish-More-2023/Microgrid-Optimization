"""
End-to-end pipeline:  python run_all.py            (full run, ~3-5 min)
                      python run_all.py --fast     (skips MPC, < 1 min)

1. Load + audit the dataset
2. Build scenarios (A: raw data, B: physics-consistent + Indian ToD tariff)
3. Forecasting benchmark (persistence / climatology / random forest)
4. Run strategies S0-S4 in every scenario
5. Pareto sweep of the battery-wear weight lambda
6. Save tables (results/tables) and figures (results/figures)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import ProjectConfig, TABLE_DIR
from src.data_audit import load_dataset, audit, save_audit
from src.scenarios import SCENARIOS
from src.forecasting import evaluate_forecasters
from src.strategies import no_battery, rule_based, lp_perfect, mpc, compute_kpis
from src.battery import assess_degradation
from src import plots


def main(fast=False):
    np.random.seed(42)
    cfg = ProjectConfig()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/6] Loading and auditing data ...")
    df = load_dataset()
    rep = audit(df)
    rep["max_possible_soc_change_pct_per_15min"] = 100 * cfg.battery.p_max_kw * 0.25 / cfg.battery.capacity_kwh
    save_audit(rep)
    print(f"      {rep['rows']} rows, {rep['start']} -> {rep['end']}")
    for f in rep["findings"]:
        print("      !", f)

    print("[2/6] Building scenarios ...")
    scenarios = {k: f(df, cfg) for k, f in SCENARIOS.items()}
    plots.plot_data_audit(df, scenarios["B_physics_ToD"])

    print("[3/6] Forecasting benchmark ...")
    fc_tables, preds = [], {}
    for name, series in [("Load (dataset)", scenarios["A_raw"].load_kw),
                         ("Renewables A (raw)", scenarios["A_raw"].ren_kw),
                         ("Renewables B (physics)", scenarios["B_physics_ToD"].ren_kw)]:
        m, p = evaluate_forecasters(series, name, seed=cfg.seed)
        fc_tables.append(m); preds[name] = p
    fc = pd.concat(fc_tables)
    fc.to_csv(TABLE_DIR / "forecast_metrics.csv", index=False)
    print(fc.round(3).to_string(index=False))
    plots.plot_forecast(fc, preds)

    print("[4/6] Running control strategies ...")
    kpis, all_results = [], {}
    for sname, sc in scenarios.items():
        print(f"   Scenario {sname}")
        runs = {
            "S0 No battery": lambda: no_battery(sc, cfg),
            "S1 Rule-based": lambda: rule_based(sc, cfg),
            "S2 LP cost-only": lambda: lp_perfect(sc, cfg, deg_weight=0.0, use_demand_charge=False),
            "S3 LP multi-objective": lambda: lp_perfect(sc, cfg, deg_weight=1.0, use_demand_charge=True),
        }
        if not fast:
            runs["S4 MPC multi-objective"] = lambda: mpc(sc, cfg, price_known=(sname != "A_raw"))
        res = {}
        for name, fn in runs.items():
            t = time.time()
            res[name] = fn()
            k = compute_kpis(name, sc, res[name], cfg); k["scenario"] = sname
            kpis.append(k)
            print(f"      {name:<24s} total cost Rs {k['total_cost_inr']:>12,.0f}   ({time.time() - t:.1f}s)")
        all_results[sname] = res
        plots.plot_dispatch(sc, res, ["S1 Rule-based", "S3 LP multi-objective"] + ([] if fast else ["S4 MPC multi-objective"]), sname)
        pd.DataFrame({k: v for k, v in res["S3 LP multi-objective"].items() if k not in ("peaks", "objective")},
                     index=sc.index).to_csv(TABLE_DIR / f"schedule_S3_{sname}.csv")

    kpi = pd.DataFrame(kpis)
    cols = ["scenario", "strategy"] + [c for c in kpi.columns if c not in ("scenario", "strategy")]
    kpi = kpi[cols]
    kpi.to_csv(TABLE_DIR / "kpi_summary.csv", index=False)
    plots.plot_cost_breakdown(kpi)
    plots.plot_soc_health(kpi)

    print("[5/6] Pareto sweep over battery-wear weight ...")
    rows = []
    for sname, sc in scenarios.items():
        for lam in [0, 0.25, 0.5, 1, 1.5, 2, 3, 5]:
            r = lp_perfect(sc, cfg, deg_weight=lam, use_demand_charge=True)
            k = compute_kpis(f"lambda={lam}", sc, r, cfg)
            k.update(scenario=sname, deg_weight=lam)
            rows.append(k)
    pareto = pd.DataFrame(rows)
    pareto.to_csv(TABLE_DIR / "pareto_sweep.csv", index=False)
    plots.plot_pareto(pareto)

    print("[6/6] Summary")
    view = kpi[["scenario", "strategy", "total_cost_inr", "energy_cost_inr", "demand_charge_inr",
                "battery_wear_cost_inr", "avg_monthly_peak_import_kw", "self_sufficiency_pct",
                "equivalent_full_cycles", "capacity_fade_pct", "projected_life_years", "co2_tonnes"]]
    print(view.round(1).to_string(index=False))
    with open(TABLE_DIR / "run_info.json", "w") as f:
        json.dump({"runtime_s": round(time.time() - t0, 1), "fast_mode": fast}, f, indent=2)
    print(f"Done in {time.time() - t0:.0f}s. Results in results/tables and results/figures.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the (slow) MPC simulation")
    main(fast=ap.parse_args().fast)
