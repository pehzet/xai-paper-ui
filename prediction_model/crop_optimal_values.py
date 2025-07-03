#!/usr/bin/env python3
"""
optimal_values.py – Erstellt Optimalwerte‑Tabellen aus *Crop_recommendation.csv*
(keine Parameter notwendig).

Ergebnisdateien
---------------
1. **optimal_ranges_long.csv**   – Long‑Format: label | parameter | opt_high | opt_low
2. **optimal_values_long.csv**   – Long‑Format: label | parameter | opt_value (Median)
3. **optimal_ranges.csv**        – **Kreuztabelle** mit Multi‑Header:
   * Zeile = Kultur (label)
   * Oberste Spaltenebene = Feature
   * Unterebene = opt_high / opt_low (max zuerst, dann min)
4. **optimal_values.csv**        – Kreuztabelle der Mediane (Feature × Kultur)

Aufruf
------
```bash
python optimal_values.py
```
Alle vier CSV‑Dateien liegen anschließend neben dem Skript.
"""

import pandas as pd
import numpy as np
from pathlib import Path
# Feste Konstanten
DATA_FILE = Path("prediction_model", "data", "Crop_recommendation.csv")
FRUIT_COL = "label"
LOWER_PCT = 0.25  # 25‑tes Perzentil → opt_low
UPPER_PCT = 0.75  # 75‑tes Perzentil → opt_high

# Output‑Dateien
OUT_RANGES_LONG = Path("prediction_model", "data", "optimal_ranges_long.csv")
OUT_VALUES_LONG = Path("prediction_model", "data", "optimal_values_long.csv")
OUT_RANGES_WIDE = Path("prediction_model", "data", "optimal_ranges.csv")   # Kombiniert opt_high & opt_low
OUT_VALUES_WIDE = Path("prediction_model", "data", "optimal_values.csv")
# Spaltennamen für Perzentile
LOW_COL = "Q1"
HIGH_COL = "Q3"

FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
def main() -> None:
    df = pd.read_csv(DATA_FILE)

    # Numerische Faktoren bestimmen
    factor_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    range_frames, value_frames = [], []

    # Long‑Format je Kultur aufbauen
    for fruit, sub in df.groupby(FRUIT_COL):
        # 25.–75. Perzentil → Q1 / Q3 (erst Q1, dann Q3)
        quant = sub[factor_cols].quantile([LOWER_PCT, UPPER_PCT]).T
        quant.columns = [LOW_COL, HIGH_COL]  # Reihenfolge bewusst Q1 vor Q3.
        quant.insert(0, FRUIT_COL, fruit)
        quant["parameter"] = quant.index
        range_frames.append(quant)

        # Median → opt_value
        med = sub[factor_cols].median().to_frame(name="opt_value")
        med.insert(0, FRUIT_COL, fruit)
        med["parameter"] = med.index
        value_frames.append(med)

    # Long‑Format DataFrames
    range_long = pd.concat(range_frames, ignore_index=True)
    values_long = pd.concat(value_frames, ignore_index=True)

    # --- Kreuztabellen ---
    # Median‑Tabelle
    values_piv = values_long.pivot(index=FRUIT_COL, columns="parameter", values="opt_value")

    # Ranges mit Multi‑Header (Feature ⟶ 10./90.)
    melt = range_long.melt(
        id_vars=[FRUIT_COL, "parameter"],
        value_vars=[LOW_COL, HIGH_COL],
        var_name="metric",
        value_name="value",
    )
    # --- Kreuztabellen ---
    melt = range_long.melt(
        id_vars=[FRUIT_COL, "parameter"],
        value_vars=[LOW_COL, HIGH_COL],
        var_name="metric",
        value_name="value",
    )

    # Reihenfolge festlegen
    metric_order  = pd.CategoricalDtype([LOW_COL, HIGH_COL], ordered=True)
    param_order   = pd.CategoricalDtype(FEATURE_ORDER, ordered=True)   # ← NEU
    melt["metric"]    = melt["metric"].astype(metric_order)
    melt["parameter"] = melt["parameter"].astype(param_order)          # ← NEU

    ranges_wide = melt.pivot_table(
        index=FRUIT_COL,
        columns=["parameter", "metric"],
        values="value",
    ).sort_index(axis=1, level=[0, 1]) 

    # --- Dateien schreiben ---
    range_long.to_csv(OUT_RANGES_LONG, index=False)
    values_long.to_csv(OUT_VALUES_LONG, index=False)
    ranges_wide.to_csv(OUT_RANGES_WIDE)
    values_piv.to_csv(OUT_VALUES_WIDE)

    print(
        "✓ Tabellen erstellt: "
        f"{OUT_RANGES_LONG}, {OUT_VALUES_LONG}, {OUT_RANGES_WIDE}, {OUT_VALUES_WIDE}"
    )


if __name__ == "__main__":
    main()
