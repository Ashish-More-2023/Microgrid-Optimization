"""
Step 1 - Data loading and quality audit.

Before optimizing anything we check whether the dataset behaves like a real
microgrid: Does solar follow the sun? Does PV follow irradiance? Do the
recorded power flows satisfy the energy balance? Is there any temporal
structure that a forecaster could exploit?  The answers decide how much the
raw data can be trusted and motivate the physics-consistent scenario.
"""
import json
import numpy as np
import pandas as pd

from .config import DATA_PATH, TABLE_DIR


def load_dataset(path=DATA_PATH, full_months_only=True):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    df.columns = [c.strip() for c in df.columns]
    if full_months_only:
        # keep only complete calendar months so monthly demand charges are fair
        counts = df.groupby(df.index.to_period("M")).size()
        expected = df.index.to_period("M").unique().map(lambda p: p.days_in_month * 96)
        full = [p for p, e in zip(counts.index, expected) if counts[p] == e]
        df = df[df.index.to_period("M").isin(full)]
    return df


def audit(df: pd.DataFrame) -> dict:
    hourly = df.groupby(df.index.hour)
    night = df.index.hour.isin([0, 1, 2, 3, 4, 22, 23])
    balance = (df.total_power_generation + df.grid_import_power + df.battery_discharge_power
               - df.battery_charge_power - df.grid_export_power - df.load_demand)
    soc_step = df.battery_soc.diff().abs()

    report = {
        "rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "missing_values": int(df.isna().sum().sum()),
        "irregular_timesteps": int((df.index.to_series().diff().dropna() != pd.Timedelta("15min")).sum()),
        "mean_solar_kw_night": float(df.loc[night, "solar_power_output"].mean()),
        "mean_solar_kw_day_11_14": float(df.loc[df.index.hour.isin([11, 12, 13]), "solar_power_output"].mean()),
        "corr_solar_vs_irradiance": float(df.solar_power_output.corr(df.solar_irradiance)),
        "corr_wind_vs_windspeed": float(df.wind_power_output.corr(df.wind_speed)),
        "load_autocorr_lag1": float(df.load_demand.autocorr(1)),
        "load_autocorr_lag96": float(df.load_demand.autocorr(96)),
        "hourly_load_profile_range_kw": float(hourly.load_demand.mean().max() - hourly.load_demand.mean().min()),
        "energy_balance_residual_mean_kw": float(balance.mean()),
        "energy_balance_residual_std_kw": float(balance.std()),
        "mean_abs_soc_jump_pct_per_15min": float(soc_step.mean()),
        "max_possible_soc_change_pct_per_15min": None,  # filled by caller with battery size
        "optimal_energy_dispatch_equals_gen_minus_load": bool(
            np.allclose(df.optimal_energy_dispatch, df.total_power_generation - df.load_demand)),
    }
    findings = []
    if abs(report["mean_solar_kw_night"] - report["mean_solar_kw_day_11_14"]) < 0.1 * report["mean_solar_kw_day_11_14"]:
        findings.append("Solar output shows no day/night cycle (night mean ~ noon mean).")
    if abs(report["corr_solar_vs_irradiance"]) < 0.1:
        findings.append("Solar power is uncorrelated with irradiance.")
    if abs(report["corr_wind_vs_windspeed"]) < 0.1:
        findings.append("Wind power is uncorrelated with wind speed.")
    if abs(report["load_autocorr_lag1"]) < 0.1:
        findings.append("Load has no autocorrelation: consecutive 15-min values are independent.")
    if report["energy_balance_residual_std_kw"] > 50:
        findings.append("Recorded generation/battery/grid flows do not satisfy the power balance.")
    if report["mean_abs_soc_jump_pct_per_15min"] > 10:
        findings.append("Recorded battery SOC jumps far faster than any physical battery could move.")
    if report["optimal_energy_dispatch_equals_gen_minus_load"]:
        findings.append("Column 'optimal_energy_dispatch' is simply generation minus load (not an optimized schedule).")
    report["findings"] = findings
    return report


def save_audit(report: dict, name="data_audit.json"):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLE_DIR / name, "w") as f:
        json.dump(report, f, indent=2)
