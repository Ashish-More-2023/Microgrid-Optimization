"""Step 7 - Figures for the report (saved to results/figures)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
COLORS = {"S0 No battery": "#9aa0a6", "S1 Rule-based": "#f4a261", "S2 LP cost-only": "#e76f51",
          "S3 LP multi-objective": "#2a9d8f", "S4 MPC multi-objective": "#264653"}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def plot_data_audit(df: pd.DataFrame, sc_phys: pd.DataFrame):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    h = df.index.hour
    ax[0].plot(df.groupby(h).solar_power_output.mean(), "o-", label="Dataset solar_power_output")
    ax[0].plot(sc_phys.groupby(h).pv_kw.mean(), "s-", label="Physics PV model (Scenario B)")
    ax[0].set(title="Mean PV output by hour", xlabel="Hour of day", ylabel="kW"); ax[0].legend()
    sample = df.sample(1500, random_state=0)
    ax[1].scatter(sample.solar_irradiance, sample.solar_power_output, s=4, alpha=0.4)
    ax[1].set(title=f"Solar power vs irradiance (r = {df.solar_power_output.corr(df.solar_irradiance):.3f})",
              xlabel="Irradiance (W/m²)", ylabel="Solar power (kW)")
    ax[2].plot(df.groupby(h).load_demand.mean(), "o-", color="#264653")
    ax[2].fill_between(range(24), df.groupby(h).load_demand.quantile(0.1), df.groupby(h).load_demand.quantile(0.9), alpha=0.2)
    ax[2].set(title="Load by hour (mean, 10–90 % band)", xlabel="Hour of day", ylabel="kW")
    _save(fig, "01_data_audit.png")


def plot_dispatch(sc, results: dict, strategies, scenario, days=3, start_day=10):
    s = start_day * 96
    sl = slice(s, s + days * 96)
    idx = sc.index[sl]
    fig, ax = plt.subplots(len(strategies) + 1, 1, figsize=(12, 2.4 * (len(strategies) + 1)), sharex=True)
    ax[0].plot(idx, sc.load_kw.iloc[sl], label="Load", color="k", lw=0.8)
    ax[0].plot(idx, sc.ren_kw.iloc[sl], label="Renewables", color="#e9c46a", lw=0.8)
    a2 = ax[0].twinx(); a2.step(idx, sc.price_import.iloc[sl], color="#e76f51", lw=0.8, where="post")
    a2.set_ylabel("Rs/kWh", color="#e76f51"); a2.grid(False)
    ax[0].set(title=f"Scenario {scenario}: inputs", ylabel="kW"); ax[0].legend(loc="upper left")
    for a, name in zip(ax[1:], strategies):
        r = results[name]
        a.plot(idx, r["p_imp"][sl], label="Grid import", color="#264653", lw=0.8)
        a.plot(idx, r["p_dis"][sl] - r["p_ch"][sl], label="Battery (+dis / −ch)", color="#2a9d8f", lw=0.8)
        b = a.twinx(); b.plot(idx, r["energy"][sl] / 10, color="#f4a261", lw=1.2); b.set_ylabel("SOC %", color="#f4a261")
        b.set_ylim(0, 100); b.grid(False)
        a.set(title=name, ylabel="kW"); a.legend(loc="upper left", fontsize=7)
    _save(fig, f"02_dispatch_{scenario}.png")


def plot_cost_breakdown(kpi: pd.DataFrame):
    scen = kpi.scenario.unique()
    fig, ax = plt.subplots(1, len(scen), figsize=(6 * len(scen), 3.8))
    ax = np.atleast_1d(ax)
    for a, s in zip(ax, scen):
        k = kpi[kpi.scenario == s].set_index("strategy")
        bottom = np.zeros(len(k))
        for col, c, lab in [("energy_cost_inr", "#264653", "Energy"), ("demand_charge_inr", "#e9c46a", "Demand charge"),
                            ("battery_wear_cost_inr", "#e76f51", "Battery wear")]:
            a.bar(range(len(k)), k[col] / 1e5, bottom=bottom, color=c, label=lab)
            bottom += k[col].to_numpy() / 1e5
        a.set_xticks(range(len(k))); a.set_xticklabels([n.replace(" ", "\n", 1) for n in k.index], fontsize=7)
        a.set(title=f"Scenario {s}: total cost (2 months)", ylabel="Rs lakh"); a.legend(fontsize=7)
    _save(fig, "03_cost_breakdown.png")


def plot_pareto(pareto: pd.DataFrame):
    scen = pareto.scenario.unique()
    fig, ax = plt.subplots(1, len(scen), figsize=(6 * len(scen), 3.8))
    ax = np.atleast_1d(ax)
    for a, s in zip(ax, scen):
        p = pareto[pareto.scenario == s]
        a.plot(p.capacity_fade_pct, (p.energy_cost_inr + p.demand_charge_inr) / 1e5, "o-", color="#2a9d8f")
        for _, r in p.iterrows():
            a.annotate(f"λ={r.deg_weight:g}", (r.capacity_fade_pct, (r.energy_cost_inr + r.demand_charge_inr) / 1e5),
                       fontsize=7, xytext=(4, 3), textcoords="offset points")
        a.set(title=f"Scenario {s}: cost vs battery wear (Pareto front)",
              xlabel="Battery capacity fade over 2 months (%)", ylabel="Energy + demand cost (Rs lakh)")
    _save(fig, "04_pareto_cost_vs_wear.png")


def plot_forecast(fc: pd.DataFrame, preds: dict):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    piv = fc.pivot(index="series", columns="model", values="R2")
    piv.plot.bar(ax=ax[0], rot=0, color=["#264653", "#2a9d8f", "#e76f51"])
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set(title="One-step forecast skill (R², test set)", ylabel="R²")
    key = list(preds)[-1]
    p = preds[key].iloc[:288]
    ax[1].plot(p.index, p.actual, color="k", lw=0.7, label="Actual")
    ax[1].plot(p.index, p.climatology, color="#2a9d8f", lw=1, label="Climatology")
    ax[1].plot(p.index, p.random_forest, color="#e76f51", lw=1, label="Random forest")
    ax[1].set(title=f"{key}: 3 test days", ylabel="kW"); ax[1].legend(fontsize=7)
    _save(fig, "05_forecasting.png")


def plot_soc_health(kpi: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for a, col, t in [(ax[0], "equivalent_full_cycles", "Equivalent full cycles (2 months)"),
                      (ax[1], "projected_life_years", "Projected battery life to 80 % SOH (years)")]:
        piv = kpi.pivot(index="strategy", columns="scenario", values=col)
        piv.plot.barh(ax=a, color=["#9aa0a6", "#2a9d8f"]); a.set(title=t, ylabel="")
    _save(fig, "06_battery_health.png")
