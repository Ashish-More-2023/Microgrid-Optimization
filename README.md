# Data-Driven Energy Management and Optimization of a Building-Scale Microgrid

Multi-objective optimization of a grid-connected solar + wind + battery microgrid that
jointly minimises **energy cost**, **peak grid demand** (equipment stress) and **battery
degradation** (equipment life), evaluated on a 15-minute microgrid dataset under an
Indian Time-of-Day (ToD) tariff.

## What the project does
| Step | Module | What happens |
|---|---|---|
| 1 | `src/data_audit.py` | Loads the dataset and checks physical consistency (day/night cycle, PV vs irradiance, power balance, SOC dynamics) |
| 2 | `src/scenarios.py` | Scenario **A** (raw dataset) and Scenario **B** (physics-based PV & wind models + Indian ToD tariff) |
| 3 | `src/forecasting.py` | Persistence vs climatology vs Random Forest forecasts |
| 4 | `src/battery.py` | Throughput degradation cost + rainflow cycle counting + Wöhler life model |
| 5 | `src/optimizer.py` | Linear-programming EMS (HiGHS solver) with cost + wear + monthly peak objectives |
| 6 | `src/strategies.py` | S0 no battery, S1 rule-based, S2 LP cost-only, S3 LP multi-objective, S4 MPC |
| 7 | `src/plots.py` | All figures for the report |

## Quick start
```bash
git clone https://github.com/<your-username>/microgrid-optimization.git
cd microgrid-optimization
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python tests/test_core.py        # sanity checks (~10 s)
python run_all.py --fast         # everything except MPC (< 1 min)
python run_all.py                # full run incl. MPC (~2-4 min)
```
Outputs appear in `results/tables/` (CSV/JSON) and `results/figures/` (PNG).

## Changing assumptions
Every parameter (battery size, efficiencies, SOC window, replacement cost, cycle life,
grid limits, feed-in tariff, demand charge, ToD rates, PV/wind ratings, MPC horizon)
is in `src/config.py`.

## Key results (Jan–Feb 2023 data, 1 MWh / 250 kW battery)
| Scenario | Strategy | Total cost (Rs lakh) | Avg monthly peak import (kW) | Projected battery life (yr) |
|---|---|---|---|---|
| B physics + ToD | S0 No battery | 31.8 | 799 | – |
| B physics + ToD | S1 Rule-based | 31.8 | 799 | 10.1 |
| B physics + ToD | S2 LP cost-only | 31.4 | 800 | 7.0 |
| B physics + ToD | S3 LP multi-objective | **29.3** | **549** | 8.7 |
| B physics + ToD | S4 MPC multi-objective | 30.8 | 772 | 8.6 |

See `results/tables/kpi_summary.csv` for all KPIs in both scenarios.

## Important note about the dataset
The data audit shows the dataset is synthetic and statistically independent from step to
step (solar output at midnight, no PV–irradiance correlation, recorded flows that do not
balance). Scenario B therefore rebuilds generation from the weather columns with physical
models. Results should be read as a methodology demonstration; re-run on measured
building data for engineering conclusions.

## References
1. M. F. Zia, E. Elbouchikhi, M. Benbouzid, "Microgrids energy management systems: A critical review on methods, solutions, and prospects," *Applied Energy*, vol. 222, pp. 1033–1055, 2018.
2. A. Parisio, E. Rikos, L. Glielmo, "A Model Predictive Control Approach to Microgrid Operation Optimization," *IEEE Trans. Control Systems Technology*, 22(5), 1813–1827, 2014.
3. A. Parisio, E. Rikos, L. Glielmo, "Stochastic model predictive control for economic/environmental operation management of microgrids: An experimental case study," *Journal of Process Control*, vol. 43, pp. 24–37, 2016.
4. B. Xu, A. Oudalov, A. Ulbig, G. Andersson, D. S. Kirschen, "Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment," *IEEE Trans. Smart Grid*, 9(2), 1131–1140, 2018.
5. M. A. Ortega-Vazquez, "Optimal scheduling of electric vehicle charging and vehicle-to-grid services at household level including battery degradation and price uncertainty," *IET Gen. Transm. Distrib.*, 8(6), 1007–1016, 2014.
6. Ministry of Power, Govt. of India, Electricity (Rights of Consumers) Amendment Rules, 2023 (Time-of-Day tariff).

## License
MIT
