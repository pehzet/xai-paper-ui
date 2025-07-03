#!/usr/bin/env python3
"""
s shap_global.py – Globale SHAP‑Feature‑Importance (Bar‑Chart) für dein
Keras/TensorFlow‑Neural‑Netz (CropPredictor) mit Multi‑Class‑Ausgabe.

Aufrufbeispiel
~~~~~~~~~~~~~~
```python
from neural_network import CropPredictor
from shap_global import plot_global_shap_bar

predictor = CropPredictor()           # bereits trainiert & mit .X_train etc.
plot_global_shap_bar(predictor)       # zeigt Diagramm & speichert PNG
```

Das Diagramm basiert auf dem **mittleren Absolutbetrag** der SHAP‑Werte,
aggregiert erst über alle Klassen, dann über alle Samples.
"""

import numpy as np
import shap
import matplotlib.pyplot as plt


def plot_global_shap_bar(
    predictor,
    background_size: int = 2000,
    sample_size: int = 2000,
    save_path: str | None = "shap_global_bar.png",
):
    """Erstellt ein Bar‑Chart der **globalen** Feature‑Wichtigkeit.

    Parameters
    ----------
    predictor       : CropPredictor
        Trainiertes Objekt mit Attributen ``model``, ``X_train``, ``X_test``
        und einer Methode ``preprocess_data(X, fit=False)``.
    background_size : int, default 100
        Anzahl zufällig gezogener Trainings­beispiele für den DeepExplainer.
    sample_size     : int, default 500
        Anzahl zufällig gezogener Test­beispiele, auf denen SHAP berechnet
        wird.  Größere Zahl = stabilere Werte, aber längere Laufzeit.
    save_path       : str | None, default "shap_global_bar.png"
        Pfad für das PNG.  ``None`` ➜ nur anzeigen, nicht speichern.
    """

    # ---------------------------------------------------------------
    # 1) Hintergrunddaten (Baseline) für DeepExplainer
    # ---------------------------------------------------------------
    bg_idx = np.random.choice(
        len(predictor.X_train),
        size=min(background_size, len(predictor.X_train)),
        replace=False,
    )
    background = predictor.preprocess_data(predictor.X_train[bg_idx], fit=False)

    explainer = shap.DeepExplainer(predictor.model, background)

    # ---------------------------------------------------------------
    # 2) Stichprobe, für die SHAP‑Werte berechnet werden
    # ---------------------------------------------------------------
    smp_idx = np.random.choice(
        len(predictor.X_test),
        size=min(sample_size, len(predictor.X_test)),
        replace=False,
    )
    X_sample = predictor.preprocess_data(predictor.X_test[smp_idx], fit=False)

    shap_values = explainer(X_sample)  # Explanation‑Objekt (N, F, C)

    # ---------------------------------------------------------------
    # 3) Globale Aggregation: |SHAP| über Klassen & Samples mitteln
    # ---------------------------------------------------------------
    shap_abs = np.abs(shap_values.values)      # (N, F, C)
    shap_agg = shap_abs.mean(axis=2)           # (N, F) – Klassen gemittelt

    # ---------------------------------------------------------------
    # 4) Bar‑Plot (summary_plot mit plot_type="bar")
    # ---------------------------------------------------------------
    shap.summary_plot(
        shap_agg,
        X_sample,
        feature_names=predictor.numerical_features,
        plot_type="bar",
        show=False,
    )
    plt.xlabel("Feature Importance")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[INFO] SHAP-Bar-Plot gespeichert unter: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Beispielaufruf (nur zu Testzwecken, normalerweise in CropPredictor integriert)
    from neural_network import CropPredictor

    predictor = CropPredictor()  # Annahme: bereits trainiert
    plot_global_shap_bar(predictor)  # Diagramm anzeigen und speichern