"""xlwings (Vollversion) + HiGHS LP/MILP Batteriesimulation."""

import time
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from typing import Any, cast

import highspy
import numpy as np
import pandas as pd
import xlwings as xw
from scipy import sparse

"""
-----------------------------------------
Datenmodelle und abgeleitete Einstellungen der Batteriesimulation.
-----------------------------------------
"""

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
}


@dataclass
class SimulationParams:
    """Simulationsparameter"""

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

    @staticmethod
    def to_dec(v: float) -> float:
        """Return a value between 0 and 1"""
        v = v or 0.0
        return v / 100.0 if v > 1.0 else v

    def set_capacity(self, capacity: float) -> None:
        """Set capacity and update all capacity-dependent parameters."""
        self.kapazitaet = float(capacity or 0)
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
        """Initiate class"""
        self.netzent = float(self.netzent or 0)
        self.marge = float(self.marge or 0)

        # Verluste in Prozent als float (z.B. 0.92)
        self.ladeeff: float = self.to_dec(self.ladeeffizienz)
        self.entladeeff: float = self.to_dec(self.entladeeffizienz)
        self.wandlung: float = 1.0 - self.to_dec(self.wandlungseffizienz)

        self.selbstentladung = 1.0 - self.to_dec(self.selbstentladung)
        self.set_capacity(self.kapazitaet)

        if self.ladeeff <= 0 or self.entladeeff <= 0 or self.wandlung <= 0:
            error_msg = (
                "Lade-, Entlade- und Wandlungseffizienz müssen größer als 0 sein."
            )
            raise ValueError(error_msg)


def create_simulation_params(values: Mapping[str, object]) -> SimulationParams:
    """Erstellt Simulationsparameter aus Excel-Werten anhand der Alias-Tabelle."""
    model_fields = {field.name: field for field in fields(SimulationParams)}
    unknown_fields: set[str] = set(SIMULATION_PARAM_ALIASES.values()) - set(
        model_fields
    )
    if unknown_fields:
        error_msg = "Unbekannte SimulationParams-Felder: " + ", ".join(
            sorted(unknown_fields)
        )
        raise ValueError(error_msg)

    missing_parameters: list[str] = [
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
        field_value: object = values[excel_name]
        field_info = model_fields[field_name]
        if field_info.default is not MISSING and (
            field_value is None
            or (isinstance(field_value, (float, np.floating)) and np.isnan(field_value))
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
    parameter: dict[str, object] = dict(
        zip(
            df_parameter.iloc[:, 0].astype(str).str.strip(),
            df_parameter.iloc[:, 1],
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
    roundtrip_batt_netz: float
    roundtrip_batt_pv: float


def create_simulation_config(params: SimulationParams) -> SimulationConfig:
    """Berechnet die aus den Eingabeparametern abgeleiteten Simulationswerte."""
    is_mispel: bool = params.system.strip().lower() == "mispel"
    is_dc_kopplung: bool = params.kopplung.strip().upper().startswith("DC")

    verl_laden_pv: float = params.ladeeff * (1 if is_dc_kopplung else params.wandlung)
    verl_laden_netz: float = params.wandlung * params.ladeeff
    verl_netzeinsp_batterie: float = params.entladeeff * params.wandlung

    return SimulationConfig(
        is_mispel=is_mispel,
        is_dc_kopplung=is_dc_kopplung,
        netzentgelt=params.netzent,
        marge=params.marge,
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
        roundtrip_batt_netz=verl_laden_netz * verl_netzeinsp_batterie,
        roundtrip_batt_pv=verl_laden_pv * verl_netzeinsp_batterie,
    )


"""Vorbereitung und Extraktion der Simulationszeitreihen."""


def prepare_timeseries_data(
    df_daten: pd.DataFrame,
) -> pd.DataFrame:
    """Trennt Datum, PV-/Last- und Preisspalten anhand ihrer Überschriften."""
    if df_daten is None or df_daten.empty:
        error_msg = "Fehler: Es wurden keine Daten übergeben. "
        raise ValueError(error_msg)

    date_columns: list[object] = [
        col
        for col in df_daten.columns
        if str(col).strip().casefold() in {"datum", "date"}
    ]
    if len(date_columns) != 1:
        error_msg = "Fehler: df_daten muss genau eine Spalte 'Datum' enthalten."
        raise ValueError(error_msg)

    date_column: object = date_columns[0]
    price_column_names: set[str] = {"börsenpreis", "vergütung", "bezugspreis"}
    price_columns: list[object] = [
        col
        for col in df_daten.columns
        if str(col).strip().casefold() in price_column_names
    ]
    pv_columns: list[object] = [
        col
        for col in df_daten.columns
        if col != date_column and col not in price_columns
    ]

    date_series: pd.Series = df_daten[date_column].rename(None)

    if len(date_series) > 0 and isinstance(date_series.iloc[0], str):
        try:
            pd.to_datetime(date_series.iloc[0])
        except Exception:
            date_series = date_series.iloc[1:]

    if pd.api.types.is_datetime64_any_dtype(date_series.dtype):
        date_series = pd.to_datetime(date_series, errors="coerce").dt.round("s")
    else:
        numeric_dates: pd.Series = pd.to_numeric(date_series, errors="coerce")
        if numeric_dates.notna().sum() > (len(date_series) * 0.5):
            date_series = pd.to_datetime(
                numeric_dates, unit="D", origin="1899-12-30", errors="coerce"
            ).dt.round("s")
        else:
            date_series = pd.to_datetime(date_series, errors="coerce")

    min_len: int = min(len(df_daten), len(date_series))
    if min_len == 0:
        error_msg = "Fehler: Datenreihen enthalten nach der Bereinigung keine Zeilen."
        raise ValueError(error_msg)

    df_pv_clean: pd.DataFrame = df_daten.loc[:, pv_columns].iloc[:min_len].copy()
    date_series_clean: pd.Series = date_series.iloc[:min_len]

    df_pv_clean.index = date_series_clean

    df_pv_m: pd.DataFrame = df_pv_clean.rename(
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
    df_preise_clean: pd.DataFrame = df_daten.loc[:, price_columns].iloc[:min_len].copy()
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
    pv_dc: np.ndarray = (
        df.get("PV_dc", pd.Series(0, index=df.index))
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )
    raw_wr: np.ndarray = (
        df.get("PV_wr", pd.Series(100.0, index=df.index))
        .fillna(100)
        .to_numpy(dtype=np.float64)
    )
    pv_wr_eff: np.ndarray = np.where(raw_wr > 1.0, raw_wr / 100.0, raw_wr)

    if "PV_ac" in df.columns:
        pv_ac: np.ndarray = df["PV_ac"].fillna(0).to_numpy(dtype=np.float64)
    else:
        pv_ac = pv_dc * pv_wr_eff

    boersenpreis: np.ndarray = (
        df.get("Börsenpreis", pd.Series(0, index=df.index))
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )
    pv_verguetung: np.ndarray = (
        df.get("Vergütung", pd.Series(0, index=df.index))
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )
    strombedarf: np.ndarray = (
        df.get("Strombedarf", pd.Series(0, index=df.index))
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )
    bezugspreis: np.ndarray = (
        df.get("Bezugspreis", pd.Series(0, index=df.index))
        .fillna(0)
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


"""
-----------------------------------------
Formatierung von Simulationsergebnissen und Kapazitätsszenarien.
-----------------------------------------
"""


@dataclass(frozen=True)
class ColumnSpec:
    """Arrayindex, Anzeigename und Ausgabeposition einer Spalte."""

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
                (
                    column
                    for column in vars(cls).values()
                    if isinstance(column, ColumnSpec)
                ),
                key=lambda column: column.output_order,
            )
        )


NUM_RESULT_COLS: int = (
    max(
        column.index
        for column in ResultColumn.output_columns()
        if column.index is not None
    )
    + 1
)


def build_result_dataframe(df_index: pd.Index, res: np.ndarray) -> pd.DataFrame:
    """Wandelt das NumPy-Ergebnisarray in ein formatiertes DataFrame um."""
    output_columns = ResultColumn.output_columns()
    cols: list[str] = [column.name for column in output_columns]
    nan_col: np.ndarray = np.full(len(res), np.nan)
    result_values: np.ndarray = np.column_stack(
        [
            nan_col if column.index is None else res[:, column.index]
            for column in output_columns
        ]
    )
    return pd.DataFrame(result_values, index=df_index, columns=cols)


def solve_window_highs(
    config: "SimulationConfig",
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
    use_milp: bool = True,
) -> dict[str, np.ndarray]:
    """
    Löst ein einzelnes Zeitfenster (z. B. 24-48h im Rolling Horizon oder 8760h global)
    als LP oder MILP mit HiGHS.
    """
    H = len(pv_dc)
    h = highspy.Highs()
    h.silent()  # Konsolenausgabe unterdrücken

    # --- Variablen-Indizes pro Zeitschritt t (0 .. H-1) ---
    # Wir legen Blöcke der Länge H an:
    #  0: pv_to_load
    #  1: pv_to_grid
    #  2: grid_to_load
    #  3: pv_to_bat
    #  4: grid_to_bat (reguläre MiSpeL-Ladung)
    #  5: emerg_grid_to_bat (Notladung unter Min-SOC inkl. Netzentgelt)
    #  6: bat_pv_to_load
    #  7: bat_grid_to_load
    #  8: bat_pv_to_grid
    #  9: bat_grid_to_grid
    # 10: soc_pv
    # 11: soc_grid
    # 12: standby_shortfall (falls SOC < Standby-Verlust)
    # Optional 13: is_charging (Binärvariable, falls use_milp=True)

    n_blocks = 14 if use_milp else 13
    num_vars = n_blocks * H

    def idx(block: int) -> np.ndarray:
        return np.arange(block * H, (block + 1) * H, dtype=np.int32)

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
    i_stby_slack = idx(12)
    i_b_charge = idx(13) if use_milp else np.empty(0, dtype=np.int32)

    inf = highspy.kHighsInf
    lb = np.zeros(num_vars, dtype=np.float64)
    ub = np.full(num_vars, inf, dtype=np.float64)
    cost = np.zeros(num_vars, dtype=np.float64)  # Wir maximieren den Gewinn!

    # --- Obere Schranken (Bounds) setzen ---
    ub[i_pv2bat] = config.ladeleistung_pv_max
    ub[i_soc_pv] = config.kapazitaet_max
    ub[i_stby_slack] = config.standby_kwh

    if config.is_mispel:
        ub[i_grid2bat] = config.ladeleistung_netz_max
        ub[i_batgrid2load] = config.entladeleistung_last_max
        ub[i_batgrid2grid] = config.entladeleistung_netz_max
        ub[i_soc_grid] = config.kapazitaet_max
    else:
        # Kein MiSpeL -> Netz-Speicherpfade sperren (außer Notladung)
        ub[i_grid2bat] = 0.0
        ub[i_batgrid2load] = 0.0
        ub[i_batgrid2grid] = 0.0
        ub[i_soc_grid] = config.kapazitaet_max  # nur falls Notladung greift

    ub[i_batpv2load] = config.entladeleistung_last_max
    ub[i_batpv2grid] = config.entladeleistung_netz_max

    # Falls Vergütung <= 0: Keine vergütete PV-Einspeisung (Abregelung aktiv)
    vg_pos = np.where(verguetung > 0.0, verguetung, 0.0)
    ub[i_pv2grid] = np.where(verguetung > 0.0, inf, 0.0)

    if use_milp:
        ub[i_b_charge] = 1.0

    # --- Zielfunktion (Gewinn in EUR, Faktor 0.01 ct -> EUR) ---
    # Hinweis: Da grid_to_load + Eigenverbrauch = konstanter Strombedarf gilt,
    # ist die Minimierung von grid_to_load * strompreis äquivalent zu deinem
    # Eigenverbrauchsvorteil (+ (pv_to_load + bat_to_load) * strompreis).
    # Wir nutzen exakt deine Gewinnformel aus calculate_hour_profit:
    marge_penalty = 0.5 * config.marge  # Optional: Mindestmarge als Zyklus-Hürde

    cost[i_pv2grid] = vg_pos * 0.01
    cost[i_batpv2grid] = (verguetung - marge_penalty) * 0.01
    cost[i_batgrid2grid] = (boersenpreis - marge_penalty) * 0.01
    cost[i_pv2load] = strompreis * 0.01
    cost[i_batpv2load] = (strompreis - marge_penalty) * 0.01
    cost[i_batgrid2load] = (strompreis - marge_penalty) * 0.01
    cost[i_grid2bat] = -(boersenpreis + marge_penalty) * 0.01
    cost[i_emerg2bat] = (
        -(boersenpreis + config.netzentgelt + 100.0) * 0.01
    )  # Hoher Strafpreis -> nur echter Notfall
    cost[i_stby_slack] = -1e4  # Slack nur aktiv, wenn Batterie physisch leer (SOC = 0)

    # Restwert des Speichers am Ende des Rolling-Horizon-Fensters (wichtig, damit
    # das Modell den Speicher in der letzten Stunde nicht blind leer räumt!):
    cost[i_soc_pv[-1]] += terminal_val_pv * 0.01
    cost[i_soc_grid[-1]] += terminal_val_grid * 0.01

    # --- Nebenbedingungen (Sparse Matrix Aufbau) ---
    rows, cols, vals = [], [], []
    row_lb, row_ub = [], []
    r_count = 0

    def add_block_rows(
        var_tuples: list[tuple[np.ndarray, np.ndarray | float]],
        lhs: np.ndarray | float,
        rhs: np.ndarray | float,
    ) -> None:
        nonlocal r_count
        r_idx = np.arange(r_count, r_count + H, dtype=np.int32)
        for c_idx, coeff in var_tuples:
            rows.append(r_idx)
            cols.append(c_idx)
            vals.append(
                np.full(H, coeff, dtype=np.float64)
                if np.ndim(coeff) == 0
                else np.asarray(coeff, dtype=np.float64)
            )
        row_lb.append(np.full(H, lhs, dtype=np.float64) if np.ndim(lhs) == 0 else lhs)
        row_ub.append(np.full(H, rhs, dtype=np.float64) if np.ndim(rhs) == 0 else rhs)
        r_count += H

    # 1. Strombedarf decken: pv2load + batpv2load + batgrid2load + grid2load == bedarf
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

    # 2. PV-Verfügbarkeit (<= erlaubt Abregelung bei negativen Preisen)
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

    # 3. Kombinierte Leistungsbegrenzungen der Batterie & Netzanschluss
    # 3a. Gesamte Entladung zum Bedarf <= entladeleistung_last_max
    add_block_rows(
        [(i_batpv2load, 1.0), (i_batgrid2load, 1.0)],
        -inf,
        config.entladeleistung_last_max,
    )
    # 3b. Gesamte Entladung ins Netz <= entladeleistung_netz_max
    add_block_rows(
        [(i_batpv2grid, 1.0), (i_batgrid2grid, 1.0)],
        -inf,
        config.entladeleistung_netz_max,
    )
    # 3c. Gesamte Batterie-Entladung (Last + Netz) <= entladeleistung_last_max (Batteriewechselrichter-Limit)
    max_dis_total = max(
        config.entladeleistung_last_max, config.entladeleistung_netz_max
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
    # 3d. Gesamte Batterie-Beladung <= max Ladeleistung
    max_chg_total = max(config.ladeleistung_pv_max, config.ladeleistung_netz_max)
    add_block_rows(
        [(i_pv2bat, 1.0), (i_grid2bat, 1.0), (i_emerg2bat, 1.0)], -inf, max_chg_total
    )

    # 4. MILP: Kein gleichzeitiges Laden und Entladen
    if use_milp:
        # Laden <= max_chg_total * b_charge
        add_block_rows(
            [(i_pv2bat, 1.0), (i_grid2bat, 1.0), (i_b_charge, -max_chg_total)],
            -inf,
            0.0,
        )
        # Entladen <= max_dis_total * (1 - b_charge)  <=> Entladen + max_dis_total * b_charge <= max_dis_total
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

    # 5. Kapazitätsgrenzen gesamt: kapazitaet_min <= soc_pv + soc_grid <= kapazitaet_max
    add_block_rows(
        [(i_soc_pv, 1.0), (i_soc_grid, 1.0)],
        config.kapazitaet_min,
        config.kapazitaet_max,
    )

    # 6. SOC-Dynamik (PV & Netz getrennt, Standby aufgeteilt: 50/50 bzw. anteilig flexibel über Gesamt-Standby)
    # Um das LP linear zu halten, buchen wir Standby standardmäßig vom PV-SOC bzw. Grid-SOC ab:
    # Wir führen die exakten Übergangsgleichungen für t = 0 .. H-1 ein:
    s_decay = config.selbstentladung
    eta_c_pv = config.verl_laden_pv * s_decay
    eta_c_grid = config.verl_laden_netz * s_decay
    inv_eta_d = (1.0 / config.verl_netzeinsp_batterie) * s_decay

    # Hilfsvariable stby_pv in [0, standby_kwh] teilen wir direkt auf:
    # Für ein schlankes Modell buchen wir Standby im Verhältnis oder vom Gesamtspeicher;
    # hier exakt getrennt: PV trägt Standby, außer PV ist leer (dann Grid über Notladung/Bestand).
    # Noch sauberer: Wir fügen eine Variable `stby_pv[t]` (0..standby_kwh) hinzu, dann trägt Grid `standby_kwh - stby_pv - stby_slack`.
    # Um keinen extra Block zu brauchen, approximieren wir Standby proportional zur Initial-Aufteilung oder fest:
    stby_pv_share = 0.0 if config.is_mispel and init_soc_pv == 0 else 1.0
    stby_grid_share = 1.0 - stby_pv_share

    # t = 0..H-1 für soc_pv:
    # soc_pv[t] - s_decay * soc_pv[t-1] - eta_c_pv * pv2bat[t] + inv_eta_d * (batpv2load[t] + batpv2grid[t]) - stby_pv_share * stby_slack[t] == rhs
    r_pv = np.arange(r_count, r_count + H, dtype=np.int32)
    rows.extend([r_pv, r_pv, r_pv, r_pv, r_pv])
    cols.extend([i_soc_pv, i_pv2bat, i_batpv2load, i_batpv2grid, i_stby_slack])
    vals.extend(
        [
            np.ones(H),
            np.full(H, -eta_c_pv),
            np.full(H, inv_eta_d),
            np.full(H, inv_eta_d),
            np.full(H, -stby_pv_share),
        ]
    )
    if H > 1:
        rows.append(r_pv[1:])
        cols.append(i_soc_pv[:-1])
        vals.append(np.full(H - 1, -s_decay))
    rhs_pv = np.full(H, -config.standby_kwh * stby_pv_share, dtype=np.float64)
    rhs_pv[0] += s_decay * init_soc_pv
    row_lb.append(rhs_pv)
    row_ub.append(rhs_pv)
    r_count += H

    # t = 0..H-1 für soc_grid:
    r_gr = np.arange(r_count, r_count + H, dtype=np.int32)
    rows.extend([r_gr, r_gr, r_gr, r_gr, r_gr, r_gr])
    cols.extend(
        [
            i_soc_grid,
            i_grid2bat,
            i_emerg2bat,
            i_batgrid2load,
            i_batgrid2grid,
            i_stby_slack,
        ]
    )
    vals.extend(
        [
            np.ones(H),
            np.full(H, -eta_c_grid),
            np.full(H, -eta_c_grid),
            np.full(H, inv_eta_d),
            np.full(H, inv_eta_d),
            np.full(H, -stby_grid_share),
        ]
    )
    if H > 1:
        rows.append(r_gr[1:])
        cols.append(i_soc_grid[:-1])
        vals.append(np.full(H - 1, -s_decay))
    rhs_gr = np.full(H, -config.standby_kwh * stby_grid_share, dtype=np.float64)
    rhs_gr[0] += s_decay * init_soc_grid
    row_lb.append(rhs_gr)
    row_ub.append(rhs_gr)
    r_count += H

    # --- Modell an HiGHS übergeben ---
    h.addCols(
        num_vars,
        cost,
        lb,
        ub,
        0,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    h.changeObjectiveSense(highspy.ObjSense.kMaximize)

    if use_milp:
        integrality = np.full(num_vars, highspy.HighsVarType.kContinuous)
        integrality[i_b_charge] = highspy.HighsVarType.kInteger
        h.changeColsIntegrality(
            num_vars, np.arange(num_vars, dtype=np.int32), integrality
        )

    mat = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(r_count, num_vars),
    )
    h.addRows(
        r_count,
        np.concatenate(row_lb),
        np.concatenate(row_ub),
        mat.nnz,
        mat.indptr.astype(np.int32),
        mat.indices.astype(np.int32),
        mat.data.astype(np.float64),
    )

    h.run()
    sol = np.array(h.getSolution().col_value, dtype=np.float64)

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


def simulate_core_highs(
    params: "SimulationParams",
    df: pd.DataFrame,
    *,
    rolling_horizon: bool = True,
    use_milp: bool = True,
    return_array_only: bool = False,
) -> np.ndarray | pd.DataFrame:
    """Führt die Batteriesimulation mit HiGHS (Rolling Horizon oder Perfect Foresight) aus."""
    (
        pv_dc,
        pv_ac,
        wr_eff,
        boersenpreis,
        verguetung,
        strombedarf,
        strompreis,
    ) = extract_timeseries_data(df)

    n = len(pv_dc)
    config = create_simulation_config(params)
    res = np.zeros((n, NUM_RESULT_COLS), dtype=np.float64)

    soc_pv = 0.0 if config.is_mispel else params.ladestand_min_kwh
    soc_grid = params.ladestand_min_kwh if config.is_mispel else 0.0

    # Durchschnittlicher Restwert einer gespeicherten kWh am Horizont-Ende,
    # damit der Solver am Ende eines 35h-Fensters die Batterie nicht für 0.01 ct entleert:
    avg_val_pv = (
        float(np.mean(np.maximum(verguetung, strompreis)))
        * config.verl_netzeinsp_batterie
    )
    avg_val_grid = float(np.mean(boersenpreis)) * config.verl_netzeinsp_batterie

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

        # Stündliche Verluste aus Energiebilanz der Batterie:
        prev_tot = np.concatenate([[s_pv_prev + s_gr_prev], s_tot[:-1]])
        energy_in = pv2b + g2b
        energy_out = bpv2l + bgr2l + bpv2g + bgr2g
        losses = np.maximum(0.0, prev_tot + energy_in - energy_out - s_tot)

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
        # Globales Optimum über den gesamten Zeitraum (Perfect Foresight)
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
            terminal_val_pv=0.0,
            terminal_val_grid=0.0,
            use_milp=use_milp,
        )
        write_slice(0, n, sol, soc_pv, soc_grid)
    else:
        # Day-Ahead Rolling Horizon (Bekanntgabe um 13:00 Uhr für den Folgetag)
        t = 0
        while t < n:
            cur_hour = int(hours[t])
            if cur_hour < 13:
                # Vor 13 Uhr: Sicht bis Ende heute (24 - cur_hour), Ausführung bis 13 Uhr
                horizon = min(n - t, 24 - cur_hour)
                commit = min(horizon, 13 - cur_hour)
            else:
                # Ab 13 Uhr: Sicht bis Ende morgen (48 - cur_hour, also 35h um 13 Uhr),
                # Ausführung für 24h bis zum nächsten Tag um 13 Uhr!
                horizon = min(n - t, 48 - cur_hour)
                commit = min(horizon, (24 - cur_hour) + 13)

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


# 1. Als UDF (Windows mit installiertem xlwings Add-in -> "Import Functions")
@xw.func
@xw.arg("df_parameter", pd.DataFrame, index=False, header=False)
@xw.arg("df_daten", pd.DataFrame, index=False, header=True)
@xw.ret(index=False, header=True)
def BATTERIE_SIMULATION_LP(
    df_parameter: pd.DataFrame,
    df_daten: pd.DataFrame,
    rolling_horizon: bool = True,
) -> pd.DataFrame:
    """UDF-Ersatz mit HiGHS LP/MILP."""
    try:
        ts_data = prepare_timeseries_data(df_daten)
        params = simulation_params_from_dataframe(df_parameter)
        # Marge standardmäßig auf 0 setzen, falls gewünscht (LP optimiert exakt):
        # params.marge = 0.0

        result_df = cast(
            "pd.DataFrame",
            simulate_core_highs(params, ts_data, rolling_horizon=rolling_horizon),
        )
        if params.kapazitaet_2 > 0:
            # In append_capacity_scenario_summary einfach simulate_core durch simulate_core_highs ersetzen!
            result_df = append_capacity_scenario_summary_highs(
                result_df,
                params,
                ts_data,
                params.kapazitaet_2,
                params.marge_2,
                rolling_horizon,
            )
        return result_df.astype(object).replace({np.nan: None})
    except Exception as e:
        return pd.DataFrame({"Fehler": [f"Fehler: {e!s}"]})


# 2. Als Button-Makro (empfohlen für große Optimierungen, läuft auf Windows & Mac)
# In Excel im VBA-Modul dem Button zuweisen:
# Sub BtnSimulieren()
#     RunPython "import main; main.run_highs_simulation()"
# End Sub
def run_highs_simulation() -> None:
    """Liest Daten aus Excel, rechnet mit HiGHS und schreibt die Tabelle direkt zurück."""
    start_time = time.perf_counter()
    book = xw.Book.caller()
    sheet_bat = book.sheets["Batterie-Simulation"]

    df_parameter: pd.DataFrame = (
        sheet_bat.range("__Parameter")
        .options(pd.DataFrame, index=False, header=False)
        .value
    )
    df_daten: pd.DataFrame = (
        sheet_bat.range("__Daten").options(pd.DataFrame, index=False, header=True).value
    )

    result_df = BATTERIE_SIMULATION_LP(df_parameter, df_daten, rolling_horizon=True)

    # Zielzelle anpassen (z.B. benannter Bereich "__Ergebnis"):
    target = sheet_bat.range("__Ergebnis")
    target.expand().clear_contents()
    target.options(index=False, header=True).value = result_df

    book.app.status_bar = (
        f"HiGHS-Optimierung abgeschlossen in {time.perf_counter() - start_time:.2f} s."
    )


# Für lokales Debugging direkt aus VS Code / PyCharm ohne Excel-Button:
if __name__ == "__main__":
    xw.Book("DeineDatei.xlsm").set_mock_caller()
    run_highs_simulation()
