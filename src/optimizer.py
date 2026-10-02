"""
Step 4 - Multi-objective linear-programming Energy Management System (EMS).

Decision variables for every time step t (dt = 0.25 h):
    P_imp  grid import            P_exp  grid export
    P_ch   battery charge         P_dis  battery discharge
    P_cur  renewable curtailment  E      battery energy at end of step
and one peak-import variable P_pk,m for every billing month m.

Objective (weighted sum of three goals):
    min  sum_t dt*(pi_t P_imp - pe_t P_exp)            -> energy COST
       + lambda * c_deg * sum_t dt*(P_ch + P_dis)       -> battery LIFE (wear)
       + c_dem * sum_m P_pk,m                           -> grid/transformer stress (peak)
Constraints:
    P_imp - P_exp - P_ch + P_dis - P_cur = Load - Ren           (power balance)
    E_t = E_{t-1} + eta_c dt P_ch - dt/eta_d P_dis               (battery dynamics)
    E_min <= E_t <= E_max,  0 <= P_ch, P_dis <= P_max            (SOC window, C-rate)
    P_imp,t <= P_pk,m(t),   P_pk,m >= running peak               (monthly peak)
    E_T >= E_0                                                   (no free energy at the end)
Solved with the open-source HiGHS solver through scipy.optimize.linprog.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .config import ProjectConfig, DT_H


def solve_dispatch(load, ren, price_imp, price_exp, month_ids, cfg: ProjectConfig,
                   e0=None, deg_weight=None, use_demand_charge=None,
                   peak_floor=None, demand_weight_scale=1.0, terminal=True):
    """Solve the EMS LP over the given horizon. Returns a dict of arrays."""
    b, g, o = cfg.battery, cfg.grid, cfg.optim
    deg_weight = o.deg_weight if deg_weight is None else deg_weight
    use_demand_charge = o.use_demand_charge if use_demand_charge is None else use_demand_charge
    e0 = b.e_init if e0 is None else e0

    load = np.asarray(load, float); ren = np.asarray(ren, float)
    T = len(load)
    months, m_idx = np.unique(np.asarray(month_ids), return_inverse=True)
    M = len(months)
    n = 6 * T + M
    I, X, C, D, U, E = (np.arange(k * T, (k + 1) * T) for k in range(6))
    PK = np.arange(6 * T, 6 * T + M)

    # ---------------- objective ----------------
    c_deg = b.deg_cost_per_kwh_throughput * max(deg_weight, 1e-4)  # tiny value avoids simultaneous ch/dis
    c = np.zeros(n)
    c[I] = DT_H * np.asarray(price_imp)
    c[X] = -DT_H * np.asarray(price_exp)
    c[C] = DT_H * c_deg
    c[D] = DT_H * c_deg
    c[U] = o.curtail_penalty * DT_H
    c[PK] = (g.demand_charge_inr_per_kw_month * demand_weight_scale) if use_demand_charge else 0.0

    # ---------------- equality constraints ----------------
    r = np.arange(T)
    # power balance
    A1 = sp.csr_matrix((np.concatenate([np.ones(T), -np.ones(T), -np.ones(T), np.ones(T), -np.ones(T)]),
                        (np.tile(r, 5), np.concatenate([I, X, C, D, U]))), shape=(T, n))
    b1 = load - ren
    # battery dynamics: E_t - E_{t-1} - eta_c dt C_t + dt/eta_d D_t = 0
    rows = np.concatenate([r, r[1:], r, r])
    cols = np.concatenate([E, E[:-1], C, D])
    vals = np.concatenate([np.ones(T), -np.ones(T - 1), -b.eta_ch * DT_H * np.ones(T), DT_H / b.eta_dis * np.ones(T)])
    A2 = sp.csr_matrix((vals, (rows, cols)), shape=(T, n))
    b2 = np.zeros(T); b2[0] = e0
    A_eq = sp.vstack([A1, A2]).tocsr()
    b_eq = np.concatenate([b1, b2])

    # ---------------- inequality: P_imp,t - P_pk,m(t) <= 0 ----------------
    A_ub = sp.csr_matrix((np.concatenate([np.ones(T), -np.ones(T)]),
                          (np.concatenate([r, r]), np.concatenate([I, PK[m_idx]]))), shape=(T, n))
    b_ub = np.zeros(T)

    # ---------------- bounds ----------------
    lb = np.zeros(n); ub = np.full(n, np.inf)
    ub[I] = g.import_limit_kw
    ub[X] = g.export_limit_kw
    ub[C] = b.p_max_kw
    ub[D] = b.p_max_kw
    ub[U] = np.maximum(ren, 0)
    lb[E] = b.e_min; ub[E] = b.e_max
    if terminal:
        lb[E[-1]] = min(max(e0, b.e_min), b.e_max)
    if peak_floor is not None:
        lb[PK] = np.minimum(np.asarray(peak_floor, float), g.import_limit_kw)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=np.column_stack([lb, ub]), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    xv = res.x
    return {
        "p_imp": xv[I], "p_exp": xv[X], "p_ch": xv[C], "p_dis": xv[D],
        "p_cur": xv[U], "energy": xv[E], "peaks": dict(zip(months, xv[PK])),
        "objective": res.fun,
    }
