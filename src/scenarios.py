"""
Step 2 - Build optimization scenarios from the dataset.

Every scenario is a DataFrame with the same columns:
    load_kw, pv_kw, wind_kw, ren_kw, price_import, price_export

Scenario A ("raw")     : uses the dataset's power and price columns exactly as given.
Scenario B ("physics") : keeps the dataset's measured load and weather drivers
                         (irradiance, temperature, wind speed) but converts them
                         into power with standard physical models and applies an
                         Indian Time-of-Day (ToD) tariff.
"""
import numpy as np
import pandas as pd

from .config import ProjectConfig


def _daylight_shape(index: pd.DatetimeIndex, sunrise: float, sunset: float) -> np.ndarray:
    """Sinusoidal clear-sky envelope between sunrise and sunset (0 at night)."""
    h = index.hour + index.minute / 60.0 + 0.125  # centre of the 15-min interval
    x = (h - sunrise) / (sunset - sunrise)
    return np.where((x > 0) & (x < 1), np.sin(np.pi * x), 0.0)


def pv_power(df, cfg: ProjectConfig) -> np.ndarray:
    """PV model: P = P_rated * G/1000 * (1 + gamma (T_cell - 25)),
    T_cell = T_amb + (NOCT - 20)/800 * G. The dataset irradiance is treated as a
    clear-sky index scaled by a day/night envelope."""
    p = cfg.pv
    g = df["solar_irradiance"].to_numpy() * _daylight_shape(df.index, p.sunrise_h, p.sunset_h)
    t_cell = df["temperature"].to_numpy() + (p.noct - 20.0) / 800.0 * g
    power = p.rated_kwp * g / 1000.0 * (1.0 + p.temp_coeff * (t_cell - 25.0))
    return np.clip(power, 0.0, p.rated_kwp)


def wind_power(df, cfg: ProjectConfig) -> np.ndarray:
    """Piecewise wind turbine power curve (cubic between cut-in and rated speed)."""
    w = cfg.wind
    v = df["wind_speed"].to_numpy()
    frac = (v ** 3 - w.cut_in ** 3) / (w.rated_speed ** 3 - w.cut_in ** 3)
    power = np.where(v < w.cut_in, 0.0,
             np.where(v < w.rated_speed, w.rated_kw * frac,
              np.where(v < w.cut_out, w.rated_kw, 0.0)))
    return power


def tod_tariff(index: pd.DatetimeIndex, cfg: ProjectConfig) -> np.ndarray:
    t = cfg.tod
    h = index.hour
    price = np.full(len(index), t.normal_rate)
    solar = (h >= t.solar_hours[0]) & (h < t.solar_hours[1])
    peak = (h >= t.peak_hours[0]) & (h < t.peak_hours[1])
    price[solar] = t.normal_rate * (1 - t.solar_discount)
    price[peak] = t.normal_rate * (1 + t.peak_premium)
    return price


def _finish(out: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    out["ren_kw"] = out["pv_kw"] + out["wind_kw"]
    # export price never above import price -> no risk-free import/export loop
    out["price_export"] = np.minimum(cfg.grid.feed_in_tariff, 0.9 * out["price_import"])
    out["hour"] = out.index.hour
    out["month"] = out.index.to_period("M").astype(str)
    return out


def build_raw_scenario(df, cfg: ProjectConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["load_kw"] = df["load_demand"]
    out["pv_kw"] = df["solar_power_output"]
    out["wind_kw"] = df["wind_power_output"]
    out["price_import"] = df["electricity_price"]
    return _finish(out, cfg)


def build_physics_scenario(df, cfg: ProjectConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["load_kw"] = df["load_demand"]
    out["pv_kw"] = pv_power(df, cfg)
    out["wind_kw"] = wind_power(df, cfg)
    out["price_import"] = tod_tariff(df.index, cfg)
    return _finish(out, cfg)


SCENARIOS = {
    "A_raw": build_raw_scenario,
    "B_physics_ToD": build_physics_scenario,
}
