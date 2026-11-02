"""Sanity tests:  python -m pytest -q   (or)   python tests/test_core.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from src.config import ProjectConfig, DT_H
from src.data_audit import load_dataset
from src.scenarios import build_physics_scenario
from src.strategies import lp_perfect, rule_based
from src.battery import rainflow

cfg = ProjectConfig()
sc = build_physics_scenario(load_dataset(), cfg).iloc[:96 * 7]


def _check(res):
    b = cfg.battery
    bal = res["p_imp"] - res["p_exp"] - res["p_ch"] + res["p_dis"] - res["p_cur"] - (sc.load_kw - sc.ren_kw).to_numpy()
    assert np.abs(bal).max() < 1e-4, "power balance violated"
    assert res["energy"].min() >= b.e_min - 1e-6 and res["energy"].max() <= b.e_max + 1e-6, "SOC window violated"
    assert res["p_ch"].max() <= b.p_max_kw + 1e-6 and res["p_dis"].max() <= b.p_max_kw + 1e-6, "C-rate violated"
    assert res["p_imp"].max() <= cfg.grid.import_limit_kw + 1e-6, "grid limit violated"


def test_lp_feasible():
    _check(lp_perfect(sc, cfg, deg_weight=1.0, use_demand_charge=True))


def test_rule_based_feasible():
    _check(rule_based(sc, cfg))


def test_rainflow_simple():
    cyc = rainflow(np.array([0.5, 0.9, 0.1, 0.9, 0.5]))
    # ASTM E1049: the 0.8-deep swing appears as two half-cycles = one full cycle
    n08 = sum(c for d, c in cyc if abs(d - 0.8) < 1e-9)
    assert abs(n08 - 1.0) < 1e-9


if __name__ == "__main__":
    test_lp_feasible(); test_rule_based_feasible(); test_rainflow_simple(); print("All tests passed")
