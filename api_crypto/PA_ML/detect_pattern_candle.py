"""
label_candles_train.py
Entraîne un k-means (SciPy) une fois pour toutes,
écrit les candle_type dans un CSV *et* sauvegarde le modèle.
"""

import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans2, whiten
import joblib                    # <-- pour la sérialisation

from crypto_forecast_ml.data_loader import load_crypto_data_custom_range

# --- paramètres ---
SYMBOL       = "BTCUSDT"
START_DATE   = "2025-06-01T09:00:00"
END_DATE     = "2025-07-16T09:00:00"
N_CLUSTERS   = 10
OUT_CSV      = f"{SYMBOL}_labeled_candles.csv"
OUT_MODEL    = "candle_kmeans_scipy.pkl"

# --- 1) chargement historique complet ---
df = load_crypto_data_custom_range(symbol=SYMBOL,
                                   start_date=START_DATE,
                                   end_date=END_DATE)

# --- 2) features (identique à ton code) ---
def compute_features(df):
    body_size   = (df['close'] - df['open']).abs()
    upper_wick  = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick  = df[['open', 'close']].min(axis=1) - df['low']
    rng         = (df['high'] - df['low']).replace(0, 1e-9)

    variation_pct = ((df['close'] - df['open']) / df['open']).fillna(0)
    df["variation_pct"] = variation_pct

    return pd.DataFrame({
        "body_size":   body_size,
        "upper_wick":  upper_wick,
        "lower_wick":  lower_wick,
        "body_ratio":  body_size / rng,
        "upper_ratio": upper_wick / rng,
        "lower_ratio": lower_wick / rng,
        "direction":   np.sign(df['close'] - df['open']),
        "volume_zscore":
            (df['volume'] - df['volume'].rolling(1_000, 1).mean())
          /  df['volume'].rolling(1_000, 1).std(ddof=0),
    }).fillna(0)

# --- 3) entraînement k-means + sauvegarde ---
feats = compute_features(df)
X     = feats.values.astype(np.float32)

# ⚠️ whiten divise chaque colonne par son écart-type global.
stds  = X.std(axis=0, ddof=0)
Xw    = X / stds

centroids, _ = kmeans2(Xw, k=N_CLUSTERS, minit="++")

# Exporter le modèle : centroids + stds
joblib.dump({"centroids": centroids, "stds": stds}, OUT_MODEL)
print(f"✅ Modèle k-means sauvegardé → {OUT_MODEL}")

# --- 4) labelliser l’historique pour analyse offline ---
from scipy.cluster.vq import vq
labels, _ = vq(Xw, centroids)
df["candle_type"] = labels
df.to_csv(OUT_CSV, index=False)
print(f"✅ CSV labellisé écrit → {OUT_CSV}")
