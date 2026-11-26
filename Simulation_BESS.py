"""xlwings Desktop (Button-Makro) + HiGHS LP/MILP Batteriesimulation inkl. Sizing."""

import contextlib
import time
from collections.abc import Mapping
from copy import copy
from dataclasses import MISSING, dataclass, fields
from typing import Any, cast

import highspy
import numpy as np
import pandas as pd
import xlwings as xw
from scipy import sparse

ANNUITY_ZERO_RATE_THRESHOLD = 1e-6
DAY_AHEAD_ANNOUNCEMENT_HOUR = 13

# =====================================================================
# 1. DATENMODELLE & SIMULATIONSKONFIGURATION
# =====================================================================

SIMULATION_PARAM_ALIASES: dict[str, str] = {
    "Batterie - Kopplung": "kopplung",
    "Batterie - Kapazität A": "kapazitaet",
    "Batterie - Kapazität B": "kapazitaet_2",
    "Batterie - Ladeeffizienz": "ladeeffizienz",
    "Batterie - Entladeeffizienz": "entladeeffizienz",
    "Batterie - Leistungsfaktor Laden": "ladeleistung",
    "Batterie - Leistungsfaktor Entladen": "entladeleistung",
    "Batterie - Ladestand Max": "ladestand_max_proz",
    "Batterie - Ladestand Min": "ladestand_min_proz",
    "Verluste - Wandlung": "wandlungseffizienz",
    "Verluste - Selbstentladung": "selbstentladung",
    "Verluste - Stand-by": "standby",
    "Netzanschluss - Bezug": "netzanschluss_bezug",
    "Netzanschluss - Einspeisen": "netzanschluss_einspeisen",
    "Vergütung - System": "system",
    "Vergütung - Mindestmarge A": "marge",
    "Vergütung - Mindestmarge B": "marge_2",
    "Strompreis - Netzentgelte": "netzent",
    # Optionale Parameter für die Kapazitätsoptimierung (Sizing):
    "Wirtschaftlichkeit - Invest A": "capex_pro_kwh",
    "Wirtschaftlichkeit - Invest B": "capex_pro_kwh_2",
    "Wirtschaftlichkeit - Betrachtungszeitraum": "betrachtungszeitraum",
    "Wirtschaftlichkeit - kalk. Zins": "kalk_zins",
    "Wirtschaftlichkeit - Max Kapazität": "max_kapazitaet",
}


@dataclass
class SimulationParams:
    """Simulations- und Investitionsparameter."""

    system: str
    netzent: float
    kopplung: str
    kapazitaet: float
    ladestand_max_proz: float
    ladestand_min_proz: float
    ladeeffizienz: float
    entladeeffizienz: float
    ladeleistung: float
    entladeleistung: float
    netzanschluss_bezug: float
    netzanschluss_einspeisen: float
    wandlungseffizienz: float
    selbstentladung: float
    standby: float
    marge: float = 0.0
    timestep_hours: float = 1.0
    kapazitaet_2: float = 0.0
    marge_2: float = -1.0
    capex_pro_kwh: float = 400.0
    betrachtungszeitraum: float = 15.0
    kalk_zins: float = 4.0
    capex_pro_kwh_2: float = 400.0
    max_kapazitaet: float = 5000.0

    @staticmethod
    def to_dec(v: float) -> float:
        """Normiert Prozentwerte (> 1) oder Dezimalwerte auf den Bereich 0..1."""
        v = v or 0.0
        return v / 100.0 if v > 1.0 else v

    def set_capacity(self, capacity: float) -> None:
        """Setzt die Kapazität und aktualisiert alle kapazitätsabhängigen Größen."""
        self.kapazitaet = float(capacity or 0.0)
        self.standby_kwh = (
            self.kapazitaet * self.to_dec(self.standby) * self.timestep_hours
        )
        self.ladeleistung_pv_kw = (
            self.kapazitaet * self.to_dec(self.ladeleistung) / self.timestep_hours
        )
        self.ladeleistung_netz_kw = min(
            self.ladeleistung_pv_kw,
            self.netzanschluss_bezug / self.timestep_hours,
        )
        self.entladeleistung_last_kw = (
            self.kapazitaet * self.to_dec(self.entladeleistung) / self.timestep_hours
        )
        self.entladeleistung_netz_kw = min(
            self.entladeleistung_last_kw,
            self.netzanschluss_einspeisen / self.timestep_hours,
        )
        self.ladestand_max_kwh = self.kapazitaet * self.to_dec(self.ladestand_max_proz)
        self.ladestand_min_kwh = self.kapazitaet * self.to_dec(self.ladestand_min_proz)

    def __post_init__(self) -> None:
        """Initialisiert die abgeleiteten Parameter nach der Instanziierung."""
        self.netzent = float(self.netzent or 0.0)
        self.marge = float(self.marge or 0.0)

        self.ladeeff: float = self.to_dec(self.ladeeffizienz)
        self.entladeeff: float = self.to_dec(self.entladeeffizienz)
        self.wandlung: float = 1.0 - self.to_dec(self.wandlungseffizienz)
        self.selbstentladung = 1.0 - self.to_dec(self.selbstentladung)
        self.set_capacity(self.kapazitaet)

        if self.ladeeff <= 0.0 or self.entladeeff <= 0.0 or self.wandlung <= 0.0:
            error_msg = (
                "Lade-, Entlade- und Wandlungseffizienz müssen größer als 0 sein."
            )
            raise ValueError(error_msg)


def create_simulation_params(values: Mapping[str, object]) -> SimulationParams:
    """Erstellt Simulationsparameter aus Excel-Werten anhand der Alias-Tabelle."""
    model_fields = {f.name: f for f in fields(SimulationParams)}
    unknown_fields = set(SIMULATION_PARAM_ALIASES.values()) - set(model_fields)
    if unknown_fields:
        error_msg = "Unbekannte SimulationParams-Felder: " + ", ".join(
            sorted(unknown_fields)
        )
        raise ValueError(error_msg)

    missing_parameters = [
        excel_name
        for excel_name, field_name in SIMULATION_PARAM_ALIASES.items()
        if model_fields[field_name].default is MISSING and excel_name not in values
    ]
    if missing_parameters:
        error_msg = "Fehlende Parameter: " + ", ".join(missing_parameters)
        raise ValueError(error_msg)

    parameters: dict[str, object] = {}
    for excel_name, field_name in SIMULATION_PARAM_ALIASES.items():
        if excel_name not in values:
            continue
        field_value = values[excel_name]
        field_info = model_fields[field_name]
        if field_info.default is not MISSING and (
            field_value is None
            or (isinstance(field_value, (float, np.floating)) and np.isnan(field_value))
            or (isinstance(field_value, str) and field_value.strip() == "")
        ):
            continue
        if field_info.type is str:
            field_value = str(field_value)
        elif field_info.type is float:
            field_value = float(str(field_value))
        parameters[field_name] = field_value

    return SimulationParams(**cast("Any", parameters))


def simulation_params_from_dataframe(df_parameter: pd.DataFrame) -> SimulationParams:
    """Erstellt Simulationsparameter aus der zweispaltigen Excel-Parametertabelle."""
    clean_df = df_parameter.dropna(how="all")
    parameter: dict[str, object] = dict(
        zip(
            clean_df.iloc[:, 0].astype(str).str.strip(),
            clean_df.iloc[:, 1],
            strict=True,
        )
    )
    return create_simulation_params(parameter)


@dataclass(frozen=True)
class SimulationConfig:
    """Abgeleitete Einstellungen für die Kernsimulation."""

    is_mispel: bool
    is_dc_kopplung: bool
    netzentgelt: float
    marge: float
    kapazitaet_max: float
    kapazitaet_min: float
    ladeleistung_pv_max: float
    ladeleistung_netz_max: float
    entladeleistung_last_max: float
    entladeleistung_netz_max: float
    standby_kwh: float
    selbstentladung: float
    verl_laden_pv: float
    verl_laden_netz: float
    verl_netzeinsp_batterie: float


def create_simulation_config(params: SimulationParams) -> SimulationConfig:
    """Berechnet die aus den Eingabeparametern abgeleiteten Simulationswerte."""
    is_mispel = params.system.strip().lower() == "mispel"
    is_dc_kopplung = params.kopplung.strip().upper().startswith("DC")

    verl_laden_pv = params.ladeeff * (1.0 if is_dc_kopplung else params.wandlung)
    verl_laden_netz = params.wandlung * params.ladeeff
    verl_netzeinsp_batterie = params.entladeeff * params.wandlung

    return SimulationConfig(
        is_mispel=is_mispel,
        is_dc_kopplung=is_dc_kopplung,
        netzentgelt=params.netzent,
        marge=max(params.marge, 0.0),
        kapazitaet_max=params.ladestand_max_kwh,
        kapazitaet_min=params.ladestand_min_kwh,
        ladeleistung_pv_max=params.ladeleistung_pv_kw,
        ladeleistung_netz_max=params.ladeleistung_netz_kw,
        entladeleistung_last_max=params.entladeleistung_last_kw,
        entladeleistung_netz_max=params.entladeleistung_netz_kw,
        standby_kwh=params.standby_kwh,
        selbstentladung=params.selbstentladung,
        verl_laden_pv=verl_laden_pv,
        verl_laden_netz=verl_laden_netz,
        verl_netzeinsp_batterie=verl_netzeinsp_batterie,
    )


# =====================================================================
# 2. VORBEREITUNG & EXTRAKTION DER ZEITREIHEN
# =====================================================================


def prepare_timeseries_data(df_daten: pd.DataFrame) -> pd.DataFrame:
    """Trennt Datum, PV-/Last- und Preisspalten anhand ihrer Überschriften."""
    if df_daten is None or df_daten.empty:
        err_msg = "Fehler: df_daten ist None oder leer."
        raise ValueError(err_msg)

    date_columns = [
        col
        for col in df_daten.columns
        if str(col).strip().casefold() in {"datum", "date"}
    ]
    if len(date_columns) != 1:
        err_msg = (
            "Fehler: df_daten muss genau eine Spalte 'Datum' enthalten, "
            f"aber {len(date_columns)} Spalten gefunden: {date_columns}"
        )
        raise ValueError(err_msg)

    date_column = date_columns[0]
    price_column_names = {"börsenpreis", "vergütung", "bezugspreis"}
    price_columns = [
        col
        for col in df_daten.columns
        if str(col).strip().casefold() in price_column_names
    ]
    pv_columns = [
        col
        for col in df_daten.columns
        if col != date_column and col not in price_columns
    ]

    date_series = df_daten[date_column].rename(None)

    if len(date_series) > 0 and isinstance(date_series.iloc[0], str):
        try:
            pd.to_datetime(date_series.iloc[0])
        except (TypeError, ValueError):
            date_series = date_series.iloc[1:]

    if pd.api.types.is_datetime64_any_dtype(date_series.dtype):
        date_series = pd.to_datetime(date_series, errors="coerce").dt.round("s")
    else:
        numeric_dates = pd.to_numeric(date_series, errors="coerce")
        if numeric_dates.notna().sum() > (len(date_series) * 0.5):
            date_series = pd.to_datetime(
                numeric_dates, unit="D", origin="1899-12-30", errors="coerce"
            ).dt.round("s")
        else:
            date_series = pd.to_datetime(date_series, errors="coerce")

    min_len = min(len(df_daten), len(date_series))
    if min_len == 0:
        err_msg = "Fehler: Datenreihen enthalten nach der Bereinigung keine Zeilen."
        raise ValueError(err_msg)

    df_pv_clean = df_daten.loc[:, pv_columns].iloc[:min_len].copy()
    date_series_clean = date_series.iloc[:min_len]
    df_pv_clean.index = date_series_clean

    df_pv_m = df_pv_clean.rename(
        columns={
            col: "PV_dc"
            if "PV (DC)" in str(col)
            else "PV_ac"
            if "PV (AC)" in str(col)
            else "PV_wr"
            if "PV WR" in str(col)
            else col
            for col in df_pv_clean.columns
        }
    )
    df_preise_clean = df_daten.loc[:, price_columns].iloc[:min_len].copy()
    df_preise_clean.index = date_series_clean
    return pd.concat([df_pv_m, df_preise_clean], axis=1)


def extract_timeseries_data(
    df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Extrahiert PV-, Last-, Börsenpreis-, Vergütungs- und Bezugspreiszeitreihen."""
    pv_dc = (
        df.get("PV_dc", pd.Series(0.0, index=df.index))
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    raw_wr = (
        df.get("PV_wr", pd.Series(100.0, index=df.index))
        .fillna(100.0)
        .to_numpy(dtype=np.float64)
    )
    pv_wr_eff = np.where(raw_wr > 1.0, raw_wr / 100.0, raw_wr)

    if "PV_ac" in df.columns:
        pv_ac = df["PV_ac"].fillna(0.0).to_numpy(dtype=np.float64)
    else:
        pv_ac = pv_dc * pv_wr_eff

    boersenpreis = (
        df.get("Börsenpreis", pd.Series(0.0, index=df.index))
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    pv_verguetung = (
        df.get("Vergütung", pd.Series(0.0, index=df.index))
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    strombedarf = (
        df.get("Strombedarf", pd.Series(0.0, index=df.index))
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    bezugspreis = (
        df.get("Bezugspreis", pd.Series(0.0, index=df.index))
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    return (
        pv_dc,
        pv_ac,
        pv_wr_eff,
        boersenpreis,
        pv_verguetung,
        strombedarf,
        bezugspreis,
    )


# =====================================================================
# 3. HIGHS LP / MILP OPTIMIERUNGSKERN (DISPATCH & SIZING)
# =====================================================================


def solve_window_highs(
    config: SimulationConfig,
    pv_dc: np.ndarray,
    pv_ac: np.ndarray,
    wr_eff: np.ndarray,
    boersenpreis: np.ndarray,
    verguetung: np.ndarray,
    strombedarf: np.ndarray,
    strompreis: np.ndarray,
    init_soc_pv: float,
    init_soc_grid: float,
    terminal_val_pv: float = 0.0,
    terminal_val_grid: float = 0.0,
    *,
    use_milp: bool = True,
) -> dict[str, np.ndarray]:
    """Löst ein Zeitfenster (Rolling Horizon oder Gesamtzeitraum)
    als LP/MILP mit HiGHS.
    """
    n_timesteps = len(pv_dc)
    highs_solver = highspy.Highs()
    highs_solver.silent()

    # Variablenblöcke je Zeitschritt (0..n_timesteps-1):
    #  0: pv_to_load          8: bat_pv_to_grid
    #  1: pv_to_grid          9: bat_grid_to_grid
    #  2: grid_to_load       10: soc_pv
    #  3: pv_to_bat          11: soc_grid
    #  4: grid_to_bat        12: stby_pv
    #  5: emerg_grid_to_bat  13: stby_grid
    #  6: bat_pv_to_load     14: stby_slack
    #  7: bat_grid_to_load   15: b_charge (optional, wenn use_milp=True)
    n_blocks = 16 if use_milp else 15
    num_vars = n_blocks * n_timesteps

    def idx(block: int) -> np.ndarray:
        return np.arange(block * n_timesteps, (block + 1) * n_timesteps, dtype=np.int32)

    i_pv2load = idx(0)
    i_pv2grid = idx(1)
    i_grid2load = idx(2)
    i_pv2bat = idx(3)
    i_grid2bat = idx(4)
    i_emerg2bat = idx(5)
    i_batpv2load = idx(6)
    i_batgrid2load = idx(7)
    i_batpv2grid = idx(8)
    i_batgrid2grid = idx(9)
    i_soc_pv = idx(10)
    i_soc_grid = idx(11)
    i_stby_pv = idx(12)
    i_stby_grid = idx(13)
    i_stby_slack = idx(14)
    i_b_charge = idx(15) if use_milp else np.empty(0, dtype=np.int32)

    inf = highspy.kHighsInf
    lb = np.zeros(num_vars, dtype=np.float64)
    ub = np.full(num_vars, inf, dtype=np.float64)
    cost = np.zeros(num_vars, dtype=np.float64)

    if config.kapazitaet_max <= 0.0:
        ub[i_pv2bat] = 0.0
        ub[i_grid2bat] = 0.0
        ub[i_emerg2bat] = 0.0
        ub[i_batpv2load] = 0.0
        ub[i_batgrid2load] = 0.0
        ub[i_batpv2grid] = 0.0
        ub[i_batgrid2grid] = 0.0
        ub[i_soc_pv] = 0.0
        ub[i_soc_grid] = 0.0
    else:
        ub[i_pv2bat] = config.ladeleistung_pv_max
        ub[i_batpv2load] = config.entladeleistung_last_max
        ub[i_batpv2grid] = config.entladeleistung_netz_max
        ub[i_soc_pv] = config.kapazitaet_max
        ub[i_soc_grid] = config.kapazitaet_max

        if config.is_mispel:
            ub[i_grid2bat] = config.ladeleistung_netz_max
            ub[i_batgrid2load] = config.entladeleistung_last_max
            ub[i_batgrid2grid] = config.entladeleistung_netz_max
        else:
            ub[i_grid2bat] = 0.0
            ub[i_batgrid2load] = 0.0
            ub[i_batgrid2grid] = 0.0

    ub[i_stby_pv] = config.standby_kwh
    ub[i_stby_grid] = config.standby_kwh
    ub[i_stby_slack] = config.standby_kwh

    vg_pos = np.where(verguetung > 0.0, verguetung, 0.0)
    ub[i_pv2grid] = np.where(verguetung > 0.0, inf, 0.0)

    if use_milp:
        ub[i_b_charge] = 1.0

    # Zielfunktion (Maximierung des Gewinns in EUR)
    m_pen = 0.5 * config.marge
    cost[i_pv2grid] = vg_pos * 0.01
    cost[i_batpv2grid] = (verguetung - m_pen) * 0.01
    cost[i_batgrid2grid] = (boersenpreis - m_pen) * 0.01
    cost[i_pv2load] = strompreis * 0.01
    cost[i_batpv2load] = (strompreis - m_pen) * 0.01
    cost[i_batgrid2load] = (strompreis - m_pen) * 0.01
    cost[i_grid2bat] = -(boersenpreis + m_pen) * 0.01
    cost[i_emerg2bat] = (
        -(np.maximum(boersenpreis, 0.0) + config.netzentgelt + 500.0) * 0.01
    )
    cost[i_stby_slack] = -1e4

    cost[i_soc_pv[-1]] += terminal_val_pv * 0.01
    cost[i_soc_grid[-1]] += terminal_val_grid * 0.01

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    row_lb: list[np.ndarray] = []
    row_ub: list[np.ndarray] = []
    r_count = 0

    def add_block_rows(
        var_tuples: list[tuple[np.ndarray, np.ndarray | float]],
        lhs: np.ndarray | float,
        rhs: np.ndarray | float,
    ) -> None:
        nonlocal r_count
        r_idx = np.arange(r_count, r_count + n_timesteps, dtype=np.int32)
        for c_idx, coeff in var_tuples:
            rows.append(r_idx)
            cols.append(c_idx)
            vals.append(
                np.full(n_timesteps, coeff, dtype=np.float64)
                if np.ndim(coeff) == 0
                else np.asarray(coeff, dtype=np.float64)
            )
        row_lb.append(
            np.full(n_timesteps, lhs, dtype=np.float64)
            if np.ndim(lhs) == 0
            else np.asarray(lhs, dtype=np.float64)
        )
        row_ub.append(
            np.full(n_timesteps, rhs, dtype=np.float64)
            if np.ndim(rhs) == 0
            else np.asarray(rhs, dtype=np.float64)
        )
        r_count += n_timesteps

    # 1. Strombedarf decken
    bedarf_clean = np.maximum(strombedarf, 0.0)
    add_block_rows(
        [
            (i_pv2load, 1.0),
            (i_batpv2load, 1.0),
            (i_batgrid2load, 1.0),
            (i_grid2load, 1.0),
        ],
        bedarf_clean,
        bedarf_clean,
    )

    # 2. PV-Verfügbarkeit (<= erlaubt Abregelung bei negativer Vergütung)
    eff_safe = np.maximum(wr_eff, 1e-6)
    if config.is_dc_kopplung:
        add_block_rows(
            [(i_pv2load, 1.0 / eff_safe), (i_pv2bat, 1.0), (i_pv2grid, 1.0 / eff_safe)],
            -inf,
            np.maximum(pv_dc, 0.0),
        )
    else:
        add_block_rows(
            [(i_pv2load, 1.0), (i_pv2bat, 1.0), (i_pv2grid, 1.0)],
            -inf,
            np.maximum(pv_ac, 0.0),
        )

    # 3. Leistungs- & Anschlussgrenzen
    max_dis_total = max(
        config.entladeleistung_last_max, config.entladeleistung_netz_max
    )
    max_chg_total = max(config.ladeleistung_pv_max, config.ladeleistung_netz_max)

    add_block_rows(
        [(i_batpv2load, 1.0), (i_batgrid2load, 1.0)],
        -inf,
        config.entladeleistung_last_max,
    )
    add_block_rows(
        [(i_batpv2grid, 1.0), (i_batgrid2grid, 1.0)],
        -inf,
        config.entladeleistung_netz_max,
    )
    add_block_rows(
        [
            (i_batpv2load, 1.0),
            (i_batgrid2load, 1.0),
            (i_batpv2grid, 1.0),
            (i_batgrid2grid, 1.0),
        ],
        -inf,
        max_dis_total,
    )
    add_block_rows([(i_pv2bat, 1.0), (i_grid2bat, 1.0)], -inf, max_chg_total)

    # 4. MILP: Kein gleichzeitiges Laden & Entladen
    if use_milp and max_chg_total > 0.0 and max_dis_total > 0.0:
        add_block_rows(
            [(i_pv2bat, 1.0), (i_grid2bat, 1.0), (i_b_charge, -max_chg_total)],
            -inf,
            0.0,
        )
        add_block_rows(
            [
                (i_batpv2load, 1.0),
                (i_batgrid2load, 1.0),
                (i_batpv2grid, 1.0),
                (i_batgrid2grid, 1.0),
                (i_b_charge, max_dis_total),
            ],
            -inf,
            max_dis_total,
        )

    # 5. Gesamte Kapazitätsgrenzen
    add_block_rows(
        [(i_soc_pv, 1.0), (i_soc_grid, 1.0)],
        config.kapazitaet_min,
        config.kapazitaet_max,
    )

    # 6. Standby-Verlustaufteilung
    add_block_rows(
        [(i_stby_pv, 1.0), (i_stby_grid, 1.0), (i_stby_slack, 1.0)],
        config.standby_kwh,
        config.standby_kwh,
    )

    # 7. SOC-Dynamik (Selbstentladung wirkt auf Bestand nach regulärer Ladung/Entladung,
    #    Notladung greift am Stundenende nach Verlusten analog zu main.py)
    s_decay = config.selbstentladung
    eta_c_pv = config.verl_laden_pv * s_decay
    eta_c_grid = config.verl_laden_netz * s_decay
    eta_c_emerg = config.verl_laden_netz
    inv_eta_d = (1.0 / config.verl_netzeinsp_batterie) * s_decay

    r_pv = np.arange(r_count, r_count + n_timesteps, dtype=np.int32)
    rows.extend([r_pv, r_pv, r_pv, r_pv, r_pv])
    cols.extend([i_soc_pv, i_pv2bat, i_batpv2load, i_batpv2grid, i_stby_pv])
    vals.extend(
        [
            np.ones(n_timesteps),
            np.full(n_timesteps, -eta_c_pv),
            np.full(n_timesteps, inv_eta_d),
            np.full(n_timesteps, inv_eta_d),
            np.ones(n_timesteps),
        ]
    )
    if n_timesteps > 1:
        rows.append(r_pv[1:])
        cols.append(i_soc_pv[:-1])
        vals.append(np.full(n_timesteps - 1, -s_decay))
    rhs_pv = np.zeros(n_timesteps, dtype=np.float64)
    rhs_pv[0] = s_decay * init_soc_pv
    row_lb.append(rhs_pv)
    row_ub.append(rhs_pv)
    r_count += n_timesteps

    r_gr = np.arange(r_count, r_count + n_timesteps, dtype=np.int32)
    rows.extend([r_gr, r_gr, r_gr, r_gr, r_gr, r_gr])
    cols.extend(
        [
            i_soc_grid,
            i_grid2bat,
            i_emerg2bat,
            i_batgrid2load,
            i_batgrid2grid,
            i_stby_grid,
        ]
    )
    vals.extend(
        [
            np.ones(n_timesteps),
            np.full(n_timesteps, -eta_c_grid),
            np.full(n_timesteps, -eta_c_emerg),
            np.full(n_timesteps, inv_eta_d),
            np.full(n_timesteps, inv_eta_d),
            np.ones(n_timesteps),
        ]
    )
    if n_timesteps > 1:
        rows.append(r_gr[1:])
        cols.append(i_soc_grid[:-1])
        vals.append(np.full(n_timesteps - 1, -s_decay))
    rhs_gr = np.zeros(n_timesteps, dtype=np.float64)
    rhs_gr[0] = s_decay * init_soc_grid
    row_lb.append(rhs_gr)
    row_ub.append(rhs_gr)
    r_count += n_timesteps

    # Modell an HiGHS übergeben
    highs_solver.addCols(
        num_vars,
        cost,
        lb,
        ub,
        0,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    highs_solver.changeObjectiveSense(highspy.ObjSense.kMaximize)

    if use_milp and max_chg_total > 0.0 and max_dis_total > 0.0:
        need_binary = (
            (boersenpreis < 0.0) | (verguetung < 0.0) | (strompreis < verguetung)
        )
        if np.any(need_binary):
            integrality = np.full(num_vars, highspy.HighsVarType.kContinuous)
            integrality[i_b_charge[need_binary]] = highspy.HighsVarType.kInteger
            highs_solver.changeColsIntegrality(
                num_vars, np.arange(num_vars, dtype=np.int32), integrality
            )

    mat = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(r_count, num_vars),
    )
    highs_solver.addRows(
        r_count,
        np.concatenate(row_lb),
        np.concatenate(row_ub),
        mat.nnz,
        mat.indptr.astype(np.int32),
        mat.indices.astype(np.int32),
        mat.data.astype(np.float64),
    )

    highs_solver.run()
    sol_values = highs_solver.getSolution().col_value
    if len(sol_values) == 0:
        err_msg = (
            f"HiGHS konnte keine Lösung finden. Status: {highs_solver.getModelStatus()}"
        )
        raise RuntimeError(err_msg)
    sol = np.array(sol_values, dtype=np.float64)

    return {
        "pv_to_load": sol[i_pv2load],
        "pv_to_grid": sol[i_pv2grid],
        "grid_to_load": sol[i_grid2load],
        "pv_to_bat": sol[i_pv2bat],
        "grid_to_bat": sol[i_grid2bat] + sol[i_emerg2bat],
        "emerg_grid_charge": sol[i_emerg2bat],
        "bat_pv_to_load": sol[i_batpv2load],
        "bat_grid_to_load": sol[i_batgrid2load],
        "bat_pv_to_grid": sol[i_batpv2grid],
        "bat_grid_to_grid": sol[i_batgrid2grid],
        "soc_pv": sol[i_soc_pv],
        "soc_grid": sol[i_soc_grid],
    }


def calculate_annuity_per_kwh(params: SimulationParams, n_steps: int) -> float:
    """Berechnet die auf die Zeitreihenlänge skalierte Kostenannuität in EUR/kWh."""
    capex = max(params.capex_pro_kwh_2, 0.0)
    years = max(params.betrachtungszeitraum, 1.0)
    z = params.to_dec(params.kalk_zins)

    if z > ANNUITY_ZERO_RATE_THRESHOLD:
        q = 1.0 + z
        annuity_factor = (z * (q**years)) / ((q**years) - 1.0)
    else:
        annuity_factor = 1.0 / years

    annual_cost_per_kwh = capex * annuity_factor
    period_years = (n_steps * params.timestep_hours) / 8760.0
    return annual_cost_per_kwh * period_years


def solve_optimal_capacity_highs(
    params: SimulationParams,
    df: pd.DataFrame,
    *,
    mip_rel_gap: float = 0.005,
) -> tuple[float, float]:
    """Wirtschaftlich optimale Batteriekapazität (kWh) und den Nettogewinn
    nach Abzug der Speicher-Annuität (EUR) in einem einzigen MILP-Durchlauf.
    """
    pv_dc, pv_ac, wr_eff, boersenpreis, verguetung, strombedarf, strompreis = (
        extract_timeseries_data(df)
    )
    n_timesteps = len(pv_dc)
    config = create_simulation_config(params)

    r_soc_max = params.to_dec(params.ladestand_max_proz)
    r_soc_min = params.to_dec(params.ladestand_min_proz)
    r_stby = params.to_dec(params.standby) * params.timestep_hours
    r_chg = params.to_dec(params.ladeleistung) / params.timestep_hours
    r_dis = params.to_dec(params.entladeleistung) / params.timestep_hours

    grid_bezug_max = params.netzanschluss_bezug / params.timestep_hours
    grid_einsp_max = params.netzanschluss_einspeisen / params.timestep_hours
    k_max = max(params.max_kapazitaet, 1.0)

    charge_big_m = max(r_chg * k_max, 1.0)
    discharge_big_m = max(r_dis * k_max, 1.0)

    highs_solver = highspy.Highs()
    highs_solver.silent()
    highs_solver.setOptionValue("mip_rel_gap", mip_rel_gap)

    n_blocks = 16
    num_vars = n_blocks * n_timesteps + 1
    i_cap = n_blocks * n_timesteps

    def idx(block: int) -> np.ndarray:
        return np.arange(block * n_timesteps, (block + 1) * n_timesteps, dtype=np.int32)

    i_pv2load = idx(0)
    i_pv2grid = idx(1)
    i_grid2load = idx(2)
    i_pv2bat = idx(3)
    i_grid2bat = idx(4)
    i_emerg2bat = idx(5)
    i_batpv2load = idx(6)
    i_batgrid2load = idx(7)
    i_batpv2grid = idx(8)
    i_batgrid2grid = idx(9)
    i_soc_pv = idx(10)
    i_soc_grid = idx(11)
    i_stby_pv = idx(12)
    i_stby_grid = idx(13)
    i_stby_slack = idx(14)
    i_b_charge = idx(15)

    inf = highspy.kHighsInf
    lb = np.zeros(num_vars, dtype=np.float64)
    ub = np.full(num_vars, inf, dtype=np.float64)
    cost = np.zeros(num_vars, dtype=np.float64)

    ub[i_cap] = k_max
    ub[i_b_charge] = 1.0
    ub[i_batpv2grid] = grid_einsp_max

    if config.is_mispel:
        ub[i_grid2bat] = grid_bezug_max
        ub[i_batgrid2grid] = grid_einsp_max
    else:
        ub[i_grid2bat] = 0.0
        ub[i_batgrid2load] = 0.0
        ub[i_batgrid2grid] = 0.0

    vg_pos = np.where(verguetung > 0.0, verguetung, 0.0)
    ub[i_pv2grid] = np.where(verguetung > 0.0, inf, 0.0)

    m_pen = 0.5 * config.marge
    cost[i_pv2grid] = vg_pos * 0.01
    cost[i_batpv2grid] = (verguetung - m_pen) * 0.01
    cost[i_batgrid2grid] = (boersenpreis - m_pen) * 0.01
    cost[i_pv2load] = strompreis * 0.01
    cost[i_batpv2load] = (strompreis - m_pen) * 0.01
    cost[i_batgrid2load] = (strompreis - m_pen) * 0.01
    cost[i_grid2bat] = -(boersenpreis + m_pen) * 0.01
    cost[i_emerg2bat] = (
        -(np.maximum(boersenpreis, 0.0) + config.netzentgelt + 500.0) * 0.01
    )
    cost[i_stby_slack] = -1e4
    cost[i_cap] = -calculate_annuity_per_kwh(params, n_timesteps)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    row_lb: list[np.ndarray] = []
    row_ub: list[np.ndarray] = []
    r_count = 0

    def add_block_rows(
        var_tuples: list[tuple[np.ndarray, np.ndarray | float]],
        lhs: np.ndarray | float,
        rhs: np.ndarray | float,
        cap_coeff: float = 0.0,
    ) -> None:
        nonlocal r_count
        r_idx = np.arange(r_count, r_count + n_timesteps, dtype=np.int32)
        for c_idx, coeff in var_tuples:
            rows.append(r_idx)
            cols.append(c_idx)
            vals.append(
                np.full(n_timesteps, coeff, dtype=np.float64)
                if np.ndim(coeff) == 0
                else np.asarray(coeff, dtype=np.float64)
            )
        if cap_coeff != 0.0:
            rows.append(r_idx)
            cols.append(np.full(n_timesteps, i_cap, dtype=np.int32))
            vals.append(np.full(n_timesteps, cap_coeff, dtype=np.float64))
        row_lb.append(
            np.full(n_timesteps, lhs, dtype=np.float64)
            if np.ndim(lhs) == 0
            else np.asarray(lhs, dtype=np.float64)
        )
        row_ub.append(
            np.full(n_timesteps, rhs, dtype=np.float64)
            if np.ndim(rhs) == 0
            else np.asarray(rhs, dtype=np.float64)
        )
        r_count += n_timesteps

    # 1. Strombedarf decken
    bedarf_clean = np.maximum(strombedarf, 0.0)
    add_block_rows(
        [
            (i_pv2load, 1.0),
            (i_batpv2load, 1.0),
            (i_batgrid2load, 1.0),
            (i_grid2load, 1.0),
        ],
        bedarf_clean,
        bedarf_clean,
    )

    # 2. PV-Bilanz
    eff_safe = np.maximum(wr_eff, 1e-6)
    if config.is_dc_kopplung:
        add_block_rows(
            [(i_pv2load, 1.0 / eff_safe), (i_pv2bat, 1.0), (i_pv2grid, 1.0 / eff_safe)],
            -inf,
            np.maximum(pv_dc, 0.0),
        )
    else:
        add_block_rows(
            [(i_pv2load, 1.0), (i_pv2bat, 1.0), (i_pv2grid, 1.0)],
            -inf,
            np.maximum(pv_ac, 0.0),
        )

    # 3. C-Raten (abhängig von Kapazität K) + feste Netzeinspeisegrenze
    add_block_rows([(i_pv2bat, 1.0), (i_grid2bat, 1.0)], -inf, 0.0, cap_coeff=-r_chg)
    add_block_rows(
        [
            (i_batpv2load, 1.0),
            (i_batgrid2load, 1.0),
            (i_batpv2grid, 1.0),
            (i_batgrid2grid, 1.0),
        ],
        -inf,
        0.0,
        cap_coeff=-r_dis,
    )
    add_block_rows([(i_batpv2grid, 1.0), (i_batgrid2grid, 1.0)], -inf, grid_einsp_max)

    # 4. Kein gleichzeitiges Laden & Entladen
    # (Big-M, Binärzwang in kritischen Preisstunden)
    need_binary = (boersenpreis < 0.0) | (verguetung < 0.0) | (strompreis < verguetung)
    add_block_rows(
        [(i_pv2bat, 1.0), (i_grid2bat, 1.0), (i_b_charge, -charge_big_m)], -inf, 0.0
    )
    add_block_rows(
        [
            (i_batpv2load, 1.0),
            (i_batgrid2load, 1.0),
            (i_batpv2grid, 1.0),
            (i_batgrid2grid, 1.0),
            (i_b_charge, discharge_big_m),
        ],
        -inf,
        discharge_big_m,
    )

    # 5. SOC-Fenster gekoppelt an K
    add_block_rows(
        [(i_soc_pv, 1.0), (i_soc_grid, 1.0)], -inf, 0.0, cap_coeff=-r_soc_max
    )
    add_block_rows([(i_soc_pv, 1.0), (i_soc_grid, 1.0)], 0.0, inf, cap_coeff=-r_soc_min)

    # 6. Standby-Verlust gekoppelt an K
    add_block_rows(
        [(i_stby_pv, 1.0), (i_stby_grid, 1.0), (i_stby_slack, 1.0)],
        0.0,
        0.0,
        cap_coeff=-r_stby,
    )

    # 7. SOC-Übergangsgleichungen (Start-SOC bei t=0 ist r_soc_min * K)
    s_decay = config.selbstentladung
    eta_c_pv = config.verl_laden_pv * s_decay
    eta_c_grid = config.verl_laden_netz * s_decay
    eta_c_emerg = config.verl_laden_netz
    inv_eta_d = (1.0 / config.verl_netzeinsp_batterie) * s_decay

    r_pv = np.arange(r_count, r_count + n_timesteps, dtype=np.int32)
    rows.extend([r_pv, r_pv, r_pv, r_pv, r_pv])
    cols.extend([i_soc_pv, i_pv2bat, i_batpv2load, i_batpv2grid, i_stby_pv])
    vals.extend(
        [
            np.ones(n_timesteps),
            np.full(n_timesteps, -eta_c_pv),
            np.full(n_timesteps, inv_eta_d),
            np.full(n_timesteps, inv_eta_d),
            np.ones(n_timesteps),
        ]
    )
    if n_timesteps > 1:
        rows.append(r_pv[1:])
        cols.append(i_soc_pv[:-1])
        vals.append(np.full(n_timesteps - 1, -s_decay))
    if not config.is_mispel and r_soc_min > 0.0:
        rows.append(np.array([r_pv[0]], dtype=np.int32))
        cols.append(np.array([i_cap], dtype=np.int32))
        vals.append(np.array([-s_decay * r_soc_min], dtype=np.float64))
    row_lb.append(np.zeros(n_timesteps, dtype=np.float64))
    row_ub.append(np.zeros(n_timesteps, dtype=np.float64))
    r_count += n_timesteps

    r_gr = np.arange(r_count, r_count + n_timesteps, dtype=np.int32)
    rows.extend([r_gr, r_gr, r_gr, r_gr, r_gr, r_gr])
    cols.extend(
        [
            i_soc_grid,
            i_grid2bat,
            i_emerg2bat,
            i_batgrid2load,
            i_batgrid2grid,
            i_stby_grid,
        ]
    )
    vals.extend(
        [
            np.ones(n_timesteps),
            np.full(n_timesteps, -eta_c_grid),
            np.full(n_timesteps, -eta_c_emerg),
            np.full(n_timesteps, inv_eta_d),
            np.full(n_timesteps, inv_eta_d),
            np.ones(n_timesteps),
        ]
    )
    if n_timesteps > 1:
        rows.append(r_gr[1:])
        cols.append(i_soc_grid[:-1])
        vals.append(np.full(n_timesteps - 1, -s_decay))
    if config.is_mispel and r_soc_min > 0.0:
        rows.append(np.array([r_gr[0]], dtype=np.int32))
        cols.append(np.array([i_cap], dtype=np.int32))
        vals.append(np.array([-s_decay * r_soc_min], dtype=np.float64))
    row_lb.append(np.zeros(n_timesteps, dtype=np.float64))
    row_ub.append(np.zeros(n_timesteps, dtype=np.float64))
    r_count += n_timesteps

    highs_solver.addCols(
        num_vars,
        cost,
        lb,
        ub,
        0,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    highs_solver.changeObjectiveSense(highspy.ObjSense.kMaximize)

    if np.any(need_binary):
        integrality = np.full(num_vars, highspy.HighsVarType.kContinuous)
        integrality[i_b_charge[need_binary]] = highspy.HighsVarType.kInteger
        highs_solver.changeColsIntegrality(
            num_vars, np.arange(num_vars, dtype=np.int32), integrality
        )

    mat = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(r_count, num_vars),
    )
    highs_solver.addRows(
        r_count,
        np.concatenate(row_lb),
        np.concatenate(row_ub),
        mat.nnz,
        mat.indptr.astype(np.int32),
        mat.indices.astype(np.int32),
        mat.data.astype(np.float64),
    )

    highs_solver.run()
    sol_values = highs_solver.getSolution().col_value
    if len(sol_values) == 0:
        err_msg = (
            f"HiGHS konnte keine Lösung finden. Status: {highs_solver.getModelStatus()}"
        )
        raise RuntimeError(err_msg)
    sol = np.array(sol_values, dtype=np.float64)
    opt_cap = float(max(sol[i_cap], 0.0))
    net_profit = float(highs_solver.getInfo().objective_function_value)
    return opt_cap, net_profit


# =====================================================================
# 4. SIMULATIONSSCHLEIFE & ERGEBNISAUFBEREITUNG
# =====================================================================


@dataclass(frozen=True)
class ColumnSpec:
    """Arrayindex, Anzeigename und Ausgabeposition einer Ergebnisspalte."""

    index: int | None
    name: str
    output_order: int


class ResultColumn:
    """Benannte Spezifikationen der Simulations-Ergebnisspalten."""

    pv_to_load = ColumnSpec(13, "PV → Bedarf", 0)
    bat_pv_to_load = ColumnSpec(14, "Batt (PV) → Bedarf", 1)
    bat_grid_to_load = ColumnSpec(15, "Batt (Netz) → Bedarf", 2)
    grid_to_load = ColumnSpec(16, "Netz → Bedarf", 3)

    spacer_1 = ColumnSpec(None, " ", 4)

    pv_to_grid = ColumnSpec(0, "PV → Netz", 5)

    spacer_2 = ColumnSpec(None, "  ", 6)

    pv_to_bat = ColumnSpec(1, "PV → Batterie", 7)
    grid_to_bat = ColumnSpec(2, "Netz → Batterie", 8)
    bat_pv_to_grid = ColumnSpec(3, "Batt (PV) → Netz", 9)
    bat_grid_to_grid = ColumnSpec(4, "Batt (Netz) → Netz", 10)

    spacer_3 = ColumnSpec(None, "   ", 11)

    soc_total = ColumnSpec(5, "Ladestand Gesamt", 12)
    soc_pv = ColumnSpec(6, "Ladestand PV", 13)
    soc_grid = ColumnSpec(7, "Ladestand Netz", 14)

    spacer_4 = ColumnSpec(None, "    ", 15)

    losses = ColumnSpec(8, "Verluste", 16)

    spacer_5 = ColumnSpec(None, "     ", 17)

    rev_pv = ColumnSpec(9, "Vergütung PV", 18)
    rev_bat_pv = ColumnSpec(10, "Vergütung Batt (PV)", 19)
    rev_bat_grid = ColumnSpec(11, "Vergütung Batt (Netz)", 20)
    cost_bat_grid = ColumnSpec(12, "Kosten Batt (Netz)", 21)

    @classmethod
    def output_columns(cls) -> tuple[ColumnSpec, ...]:
        """Liefert alle Spalten nach ihrer Ausgabeposition sortiert."""
        return tuple(
            sorted(
                (c for c in vars(cls).values() if isinstance(c, ColumnSpec)),
                key=lambda c: c.output_order,
            )
        )


NUM_RESULT_COLS: int = (
    max(c.index for c in ResultColumn.output_columns() if c.index is not None) + 1
)


def build_result_dataframe(df_index: pd.Index, res: np.ndarray) -> pd.DataFrame:
    """Wandelt das NumPy-Ergebnisarray in ein formatiertes DataFrame um."""
    output_columns = ResultColumn.output_columns()
    cols = [column.name for column in output_columns]
    nan_col = np.full(len(res), np.nan)
    result_values = np.column_stack(
        [
            nan_col if column.index is None else res[:, column.index]
            for column in output_columns
        ]
    )
    return pd.DataFrame(result_values, index=df_index, columns=cols)


def simulate_core_highs(
    params: SimulationParams,
    df: pd.DataFrame,
    *,
    rolling_horizon: bool = True,
    use_milp: bool = True,
    return_array_only: bool = False,
) -> np.ndarray | pd.DataFrame:
    """Führt die Batteriesimulation mit HiGHS (Rolling Horizon oder global) aus."""
    pv_dc, pv_ac, wr_eff, boersenpreis, verguetung, strombedarf, strompreis = (
        extract_timeseries_data(df)
    )
    n = len(pv_dc)
    config = create_simulation_config(params)
    res = np.zeros((n, NUM_RESULT_COLS), dtype=np.float64)

    soc_pv = 0.0 if config.is_mispel else params.ladestand_min_kwh
    soc_grid = params.ladestand_min_kwh if config.is_mispel else 0.0

    # Durchschnittlicher Restwert je kWh am Ende eines Rolling-Horizon-Fensters
    avg_val_pv = (
        float(np.mean(np.maximum(verguetung, strompreis)))
        * config.verl_netzeinsp_batterie
    )
    avg_val_grid = (
        float(np.mean(np.maximum(boersenpreis, 0.0))) * config.verl_netzeinsp_batterie
    )
    hours = pd.DatetimeIndex(df.index).hour.to_numpy()

    def write_slice(
        start: int,
        commit_len: int,
        sol: dict[str, np.ndarray],
        s_pv_prev: float,
        s_gr_prev: float,
    ) -> None:
        sl = slice(start, start + commit_len)
        c_vg = verguetung[sl]
        c_bp = boersenpreis[sl]

        pv2g = np.where(c_vg > 0.0, sol["pv_to_grid"][:commit_len], 0.0)
        pv2b = sol["pv_to_bat"][:commit_len]
        g2b = sol["grid_to_bat"][:commit_len]
        emerg = sol["emerg_grid_charge"][:commit_len]
        bpv2g = sol["bat_pv_to_grid"][:commit_len]
        bgr2g = sol["bat_grid_to_grid"][:commit_len]
        pv2l = sol["pv_to_load"][:commit_len]
        bpv2l = sol["bat_pv_to_load"][:commit_len]
        bgr2l = sol["bat_grid_to_load"][:commit_len]
        g2l = sol["grid_to_load"][:commit_len]
        s_pv = sol["soc_pv"][:commit_len]
        s_gr = sol["soc_grid"][:commit_len]
        s_tot = s_pv + s_gr

        prev_tot = np.concatenate([[s_pv_prev + s_gr_prev], s_tot[:-1]])
        losses = np.maximum(
            0.0,
            prev_tot + (pv2b + g2b) - (bpv2l + bgr2l + bpv2g + bgr2g) - s_tot,
        )

        res[sl, ResultColumn.pv_to_grid.index] = pv2g
        res[sl, ResultColumn.pv_to_bat.index] = pv2b
        res[sl, ResultColumn.grid_to_bat.index] = g2b
        res[sl, ResultColumn.bat_pv_to_grid.index] = bpv2g
        res[sl, ResultColumn.bat_grid_to_grid.index] = bgr2g
        res[sl, ResultColumn.soc_total.index] = s_tot
        res[sl, ResultColumn.soc_pv.index] = s_pv
        res[sl, ResultColumn.soc_grid.index] = s_gr
        res[sl, ResultColumn.losses.index] = losses
        res[sl, ResultColumn.rev_pv.index] = pv2g * c_vg * 0.01
        res[sl, ResultColumn.rev_bat_pv.index] = bpv2g * c_vg * 0.01
        res[sl, ResultColumn.rev_bat_grid.index] = bgr2g * c_bp * 0.01
        res[sl, ResultColumn.cost_bat_grid.index] = (
            g2b - emerg
        ) * c_bp * 0.01 + emerg * (c_bp + config.netzentgelt) * 0.01
        res[sl, ResultColumn.pv_to_load.index] = pv2l
        res[sl, ResultColumn.bat_pv_to_load.index] = bpv2l
        res[sl, ResultColumn.bat_grid_to_load.index] = bgr2l
        res[sl, ResultColumn.grid_to_load.index] = g2l

    if not rolling_horizon:
        sol = solve_window_highs(
            config,
            pv_dc,
            pv_ac,
            wr_eff,
            boersenpreis,
            verguetung,
            strombedarf,
            strompreis,
            soc_pv,
            soc_grid,
            0.0,
            0.0,
            use_milp=use_milp,
        )
        write_slice(0, n, sol, soc_pv, soc_grid)
    else:
        t = 0
        while t < n:
            cur_hour = int(hours[t])
            if cur_hour < DAY_AHEAD_ANNOUNCEMENT_HOUR:
                horizon = min(n - t, 24 - cur_hour)
                commit = min(horizon, DAY_AHEAD_ANNOUNCEMENT_HOUR - cur_hour)
            else:
                # Day-Ahead Bekanntgabe um 13:00 Uhr
                # (35h Fenster, 24h Ausführung bis morgen 13:00 Uhr)
                horizon = min(n - t, 48 - cur_hour)
                commit = min(horizon, (24 - cur_hour) + DAY_AHEAD_ANNOUNCEMENT_HOUR)

            end = t + horizon
            is_last = end == n
            sol = solve_window_highs(
                config,
                pv_dc[t:end],
                pv_ac[t:end],
                wr_eff[t:end],
                boersenpreis[t:end],
                verguetung[t:end],
                strombedarf[t:end],
                strompreis[t:end],
                soc_pv,
                soc_grid,
                terminal_val_pv=0.0 if is_last else avg_val_pv,
                terminal_val_grid=0.0 if is_last else avg_val_grid,
                use_milp=use_milp,
            )
            write_slice(t, commit, sol, soc_pv, soc_grid)
            soc_pv = float(sol["soc_pv"][commit - 1])
            soc_grid = float(sol["soc_grid"][commit - 1])
            t += commit

    if return_array_only:
        return res
    return build_result_dataframe(df.index, res)


def append_capacity_scenario_b(
    result_df: pd.DataFrame,
    params: SimulationParams,
    ts_data: pd.DataFrame,
    capacity: float,
    marge: float | None = None,
    *,
    rolling_horizon: bool = True,
    use_milp: bool = True,
) -> pd.DataFrame:
    """Führt Kapazitätsszenario B aus und hängt dessen Ergebnisspalten rechts an."""
    scenario_params = copy(params)
    scenario_params.set_capacity(capacity)
    if marge is not None:
        scenario_params.marge = float(marge)

    res = cast(
        "np.ndarray",
        simulate_core_highs(
            scenario_params,
            ts_data,
            rolling_horizon=rolling_horizon,
            use_milp=use_milp,
            return_array_only=True,
        ),
    )

    summary = pd.DataFrame(
        {
            "B: PV → Bedarf": res[:, ResultColumn.pv_to_load.index],
            "B: Batt (PV) → Bedarf": res[:, ResultColumn.bat_pv_to_load.index],
            "B: Batt (Netz) → Bedarf": res[:, ResultColumn.bat_grid_to_load.index],
            "B: Netz → Bedarf": res[:, ResultColumn.grid_to_load.index],
            " ": np.nan,
            "B: PV → Netz": res[:, ResultColumn.pv_to_grid.index],
            "  ": np.nan,
            "B: PV → Batterie": res[:, ResultColumn.pv_to_bat.index],
            "B: Netz → Batterie": res[:, ResultColumn.grid_to_bat.index],
            "B: Batt (PV) → Netz": res[:, ResultColumn.bat_pv_to_grid.index],
            "B: Batt (Netz) → Netz": res[:, ResultColumn.bat_grid_to_grid.index],
            "   ": np.nan,
            "B: Ladestand Gesamt": res[:, ResultColumn.soc_total.index],
            "B: Ladestand PV": res[:, ResultColumn.soc_pv.index],
            "B: Ladestand Netz": res[:, ResultColumn.soc_grid.index],
            "    ": np.nan,
            "B: Verluste": res[:, ResultColumn.losses.index],
            "     ": np.nan,
            "B: Vergütung PV": res[:, ResultColumn.rev_pv.index],
            "B: Vergütung Batt (PV)": res[:, ResultColumn.rev_bat_pv.index],
            "B: Vergütung Batt (Netz)": res[:, ResultColumn.rev_bat_grid.index],
            "B: Kosten Batt (Netz)": res[:, ResultColumn.cost_bat_grid.index],
        },
        index=result_df.index,
    )

    separator = pd.DataFrame({"": np.nan}, index=result_df.index)
    return pd.concat([result_df, separator, summary], axis=1)


# =====================================================================
# 5. BUTTON-MAKROS FÜR EXCEL (__Sim & KAPAZITÄTSOPTIMIERUNG)
# =====================================================================


def _write_simulation_to_sheet(
    sheet_bat: xw.Sheet,
    params: SimulationParams,
    ts_data: pd.DataFrame,
) -> None:
    """Rechnet Szenario A (+ optional B) und schreibt die Ergebnisse ab Zelle __Sim."""
    app = sheet_bat.book.app
    target_cell = sheet_bat.range("__Sim")[0, 0]

    result_df = cast(
        "pd.DataFrame",
        simulate_core_highs(params, ts_data, rolling_horizon=True, use_milp=True),
    )

    marge_2: float | None = (
        None if pd.isna(params.marge_2) or params.marge_2 < 0 else params.marge_2
    )
    if params.kapazitaet_2 > 0:
        result_df = append_capacity_scenario_b(
            result_df,
            params,
            ts_data,
            params.kapazitaet_2,
            marge_2,
            rolling_horizon=True,
            use_milp=True,
        )

    app.screen_updating = False
    try:
        n_rows = len(result_df) + 1
        sheet_bat.range(target_cell, target_cell.offset(n_rows, 46)).clear_contents()

        clean_df = result_df.astype(object).where(pd.notna(result_df), None)
        target_cell.options(index=False, header=True).value = clean_df
    finally:
        app.screen_updating = True


def run_simulation() -> None:
    """Excel-Button 1: Führt die Rolling-Horizon-Simulation aus
    und schreibt ab __Sim.
    """
    start_time = time.perf_counter()
    book = xw.Book.caller()
    app = book.app
    sheet_bat = book.sheets["Batterie-Simulation"]

    app.status_bar = "Batteriesimulation (HiGHS) läuft..."
    df_parameter: pd.DataFrame = (
        sheet_bat.range("__Parameter")
        .options(pd.DataFrame, index=False, header=False)
        .value
    )
    df_daten: pd.DataFrame = (
        sheet_bat.range("__Daten").options(pd.DataFrame, index=False, header=True).value
    )

    ts_data = prepare_timeseries_data(df_daten)
    params = simulation_params_from_dataframe(df_parameter)

    _write_simulation_to_sheet(sheet_bat, params, ts_data)

    elapsed = time.perf_counter() - start_time
    app.status_bar = f"HiGHS-Simulation fertig in {elapsed:.2f} s."


def run_capacity_optimization() -> None:
    """Excel-Button 2: Bestimmt die wirtschaftlich optimale Batteriekapazität (kWh),
    trägt sie optional in _Kapazitaet_B ein und rechnet sie direkt als Szenario B mit.
    """
    start_time = time.perf_counter()
    book = xw.Book.caller()
    app = book.app
    sheet_bat = book.sheets["Batterie-Simulation"]
    sheet_ein = book.sheets["Eingabe"]

    app.status_bar = "Optimale Batteriekapazität wird mit HiGHS berechnet..."
    df_parameter: pd.DataFrame = (
        sheet_bat.range("__Parameter")
        .options(pd.DataFrame, index=False, header=False)
        .value
    )
    df_daten: pd.DataFrame = (
        sheet_bat.range("__Daten").options(pd.DataFrame, index=False, header=True).value
    )

    ts_data = prepare_timeseries_data(df_daten)
    params = simulation_params_from_dataframe(df_parameter)

    opt_cap, _ = solve_optimal_capacity_highs(params, ts_data)
    opt_cap_rounded = round(opt_cap, 1)

    with contextlib.suppress(Exception):
        sheet_ein.range("_Kapazitaet_B").value = opt_cap_rounded

    params.kapazitaet_2 = opt_cap_rounded
    params.marge_2 = 0.0
    _write_simulation_to_sheet(sheet_bat, params, ts_data)

    elapsed = time.perf_counter() - start_time
    annuity_yr = calculate_annuity_per_kwh(
        params, round(8760.0 / params.timestep_hours)
    )
    app.status_bar = (
        f"Sizing fertig ({elapsed:.2f} s): Optimale Kapazität = "
        f"{opt_cap_rounded:.1f} kWh (Annuität: {annuity_yr:.1f} €/kWh/a)."
    )


if __name__ == "__main__":
    # Zum direkten Ausführen/Debuggen aus der IDE (Arbeitsmappe muss geöffnet sein):
    xw.Book("DeineMappe.xlsm").set_mock_caller()
    run_simulation()
