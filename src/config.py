"""
Central configuration for the microgrid optimization project.

All physical, economic and battery-ageing parameters live here so that every
experiment is reproducible and every assumption is visible in one place.
Units: power in kW, energy in kWh, money in INR (Rs), time step in hours.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "microgrid_energy_dataset.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

DT_H = 0.25  # 15-minute resolution of the dataset


@dataclass
class BatteryConfig:
    capacity_kwh: float = 1000.0      # usable nameplate energy
    p_max_kw: float = 250.0           # 0.25C charge/discharge limit (protects cells)
    eta_ch: float = 0.95              # charging efficiency
    eta_dis: float = 0.95             # discharging efficiency
    soc_min: float = 0.10             # lower SOC window (limits depth of discharge)
    soc_max: float = 0.90             # upper SOC window (avoids high-SOC stress)
    soc_init: float = 0.50
    # --- ageing / economics (LFP-type assumptions, editable) ---
    replacement_cost_inr_per_kwh: float = 18000.0
    cycle_life_ref: float = 6000.0    # full cycles to end-of-life at reference DoD
    dod_ref: float = 0.80             # reference depth of discharge for cycle_life_ref
    woehler_exponent: float = 1.3     # N(d) = N_ref * (d / d_ref) ** (-k)
    eol_fade: float = 0.20            # end of life = 20 % capacity loss (80 % SOH)
    calendar_fade_per_year: float = 0.015  # assumed calendar ageing

    @property
    def e_min(self):
        return self.soc_min * self.capacity_kwh

    @property
    def e_max(self):
        return self.soc_max * self.capacity_kwh

    @property
    def e_init(self):
        return self.soc_init * self.capacity_kwh

    @property
    def deg_cost_per_kwh_throughput(self):
        """Linear throughput-based degradation cost (Rs per kWh charged OR discharged).
        Total replacement cost spread over the lifetime throughput
        2 * N_ref * DoD_ref * E (charge + discharge energy)."""
        return self.replacement_cost_inr_per_kwh / (2.0 * self.cycle_life_ref * self.dod_ref)


@dataclass
class GridConfig:
    import_limit_kw: float = 800.0    # connection / transformer limit
    export_limit_kw: float = 400.0    # export permission limit
    feed_in_tariff: float = 3.0       # Rs/kWh paid for exported energy
    demand_charge_inr_per_kw_month: float = 350.0  # contract-demand style charge
    emission_factor_kg_per_kwh: float = 0.71       # approximate Indian grid factor (assumption)


@dataclass
class PVConfig:
    rated_kwp: float = 500.0
    temp_coeff: float = -0.004        # per degC
    noct: float = 45.0                # nominal operating cell temperature, degC
    sunrise_h: float = 6.0
    sunset_h: float = 18.0


@dataclass
class WindConfig:
    rated_kw: float = 300.0
    cut_in: float = 3.0
    rated_speed: float = 12.0
    cut_out: float = 25.0


@dataclass
class ToDTariffConfig:
    """Indian Time-of-Day tariff structure following the Electricity (Rights of
    Consumers) Amendment Rules, 2023: solar hours (8 h) >= 20 % cheaper,
    peak hours >= 1.2x the normal tariff for C&I consumers, peak <= solar hours."""
    normal_rate: float = 8.0
    solar_discount: float = 0.20
    peak_premium: float = 0.20
    solar_hours: tuple = (9, 17)      # [start, end) -> 8 hours
    peak_hours: tuple = (18, 22)      # [start, end) -> 4 hours


@dataclass
class OptimConfig:
    deg_weight: float = 1.0           # lambda multiplying the degradation cost
    use_demand_charge: bool = True
    mpc_horizon_steps: int = 96       # 24 h look-ahead
    curtail_penalty: float = 0.0


@dataclass
class ProjectConfig:
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    pv: PVConfig = field(default_factory=PVConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    tod: ToDTariffConfig = field(default_factory=ToDTariffConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    seed: int = 42
