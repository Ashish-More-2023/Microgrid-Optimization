"""
Step 3 - Battery ageing (equipment-life) model.

Two complementary views, following the literature:
  * Inside the optimizer: a LINEAR throughput cost (Rs per kWh cycled), the
    standard way to price degradation in an LP (cf. Ortega-Vazquez, 2014).
  * After the optimizer: a cycle-depth-aware assessment using RAINFLOW counting
    of the SOC profile and a Woehler (DoD vs cycle life) curve, the approach used
    in semi-empirical cell-life models (cf. Xu et al., IEEE Trans. Smart Grid 2018).
"""
import numpy as np

from .config import BatteryConfig, DT_H


def _turning_points(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x
    # drop flat segments
    keep = np.concatenate(([True], np.diff(x) != 0))
    x = x[keep]
    if len(x) < 3:
        return x
    d = np.diff(x)
    sign_change = np.sign(d[1:]) != np.sign(d[:-1])
    idx = np.concatenate(([0], np.where(sign_change)[0] + 1, [len(x) - 1]))
    return x[idx]


def rainflow(soc: np.ndarray):
    """ASTM E1049 three-point rainflow counting.
    Returns a list of (depth, count) with count = 1.0 (full) or 0.5 (half)."""
    pts = _turning_points(soc)
    stack, cycles = [], []
    for p in pts:
        stack.append(p)
        while len(stack) >= 3:
            x = abs(stack[-1] - stack[-2])
            y = abs(stack[-2] - stack[-3])
            if x < y:
                break
            if len(stack) == 3:
                cycles.append((y, 0.5))
                stack.pop(0)
            else:
                cycles.append((y, 1.0))
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    for i in range(len(stack) - 1):
        cycles.append((abs(stack[i + 1] - stack[i]), 0.5))
    return cycles


def cycle_life(depth, b: BatteryConfig):
    depth = np.maximum(depth, 1e-6)
    return b.cycle_life_ref * (depth / b.dod_ref) ** (-b.woehler_exponent)


def assess_degradation(energy_kwh: np.ndarray, p_ch: np.ndarray, p_dis: np.ndarray,
                       b: BatteryConfig) -> dict:
    """Evaluate battery wear for a realized schedule."""
    soc = np.asarray(energy_kwh) / b.capacity_kwh
    cycles = rainflow(soc)
    depths = np.array([c[0] for c in cycles]) if cycles else np.array([0.0])
    counts = np.array([c[1] for c in cycles]) if cycles else np.array([0.0])
    # each full cycle of depth d consumes 1/N(d) of the life, i.e. eol_fade/N(d) capacity
    cyc_fade = float(np.sum(counts * b.eol_fade / cycle_life(depths, b))) if depths.sum() > 0 else 0.0

    days = len(soc) * DT_H / 24.0
    cal_fade = b.calendar_fade_per_year * days / 365.0
    total_fade = cyc_fade + cal_fade
    fade_per_year = total_fade * 365.0 / days if days > 0 else 0.0
    throughput = float(np.sum(p_ch + p_dis) * DT_H)
    efc = throughput / (2 * b.capacity_kwh)  # equivalent full cycles
    return {
        "throughput_kwh": throughput,
        "equivalent_full_cycles": efc,
        "rainflow_cycles": float(counts.sum()),
        "mean_cycle_depth_pct": float(100 * np.average(depths, weights=counts)) if counts.sum() > 0 else 0.0,
        "cycle_fade_pct": 100 * cyc_fade,
        "calendar_fade_pct": 100 * cal_fade,
        "capacity_fade_pct": 100 * total_fade,
        "projected_life_years": (b.eol_fade / fade_per_year) if fade_per_year > 0 else np.inf,
        "rainflow_degradation_cost_inr": cyc_fade / b.eol_fade * b.replacement_cost_inr_per_kwh * b.capacity_kwh,
    }
