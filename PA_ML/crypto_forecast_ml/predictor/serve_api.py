# predictor/serve_api.py
import logger
from fastapi import FastAPI, Query
from PA_ML.crypto_forecast_ml.data_loader import load_crypto_data
from PA_ML.crypto_forecast_ml.features.technical_indicators import add_technical_indicators
from PA_ML.crypto_forecast_ml.features.target_builder import build_targets
from PA_ML.crypto_forecast_ml.predictor.predict import predict_direction
from PA_ML.crypto_forecast_ml.data_loader import load_crypto_data_custom_range
from PA_ML.candlestick_patterns import detect_classic_patterns

import traceback
app = FastAPI()
import logging
import traceback
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans2, whiten
import os
import joblib
from scipy.cluster.vq import vq

# ✅ Initialise le logger proprement
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/predict-latest")
def predict_latest(symbol: str = Query("BTCUSDT", description="Crypto symbol")):
    try:
        logger.info(f"🔵 API called with symbol: {symbol}")
        df = load_crypto_data(symbol=symbol, days=3)
        df = add_technical_indicators(df)
        df = build_targets(df)
        df_pred = predict_direction(df)

        result = df_pred.tail(10).to_dict(orient="records")
        return {"symbol": symbol, "predictions": result}

    except Exception as e:
        logger.error("🔥 Exception occurred:")
        traceback.print_exc()  # Affiche la stack trace complète
        return {"error": str(e)}




FMT_MIN = "%Y-%m-%dT%H:%M"      # 2025-07-14T13:03 (UTC)

def parse_min(dt_str: str) -> datetime:
    """Parse YYYY-MM-DDTHH:MM as UTC-aware datetime."""
    return datetime.strptime(dt_str, FMT_MIN).replace(tzinfo=timezone.utc)        # 2025-07-06T09:15

@app.get("/load-data")
def load_data(
    symbol: str = Query(...),
    start_date: str = Query(..., description="YYYY-MM-DDTHH:MM 🇫🇷"),
    end_date:   str = Query(..., description="YYYY-MM-DDTHH:MM 🇫🇷")
):
    # 👉 parse + convert -> UTC
    start_utc = parse_min(start_date)

    end_utc   = parse_min(end_date)

    df = load_crypto_data_custom_range(symbol=symbol,
                                       start_date=start_utc,
                                       end_date=end_utc)

    return {
        "symbol": symbol,
        "start_date": start_utc.isoformat(),
        "end_date":   end_utc.isoformat(),
        "data": df.to_dict(orient="records")
    }

def load_patterns(filename="patterns_significatifs.csv"):
    base_dir = os.path.dirname(__file__)
    pattern_path = os.path.join(base_dir, filename)
    patterns_df = pd.read_csv(pattern_path)

    return {
        tuple(eval(row["sequence"])): {
            "bias": row["bias"],
            "bullish_ratio": row["bullish_ratio"],
            "bearish_ratio": row["bearish_ratio"],
        }
        for _, row in patterns_df.iterrows()
    }

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


def assign_candle_types(df):
    filename = "candle_kmeans_scipy.pkl"
    base_dir = os.path.dirname(__file__)
    pattern_path = os.path.join(base_dir, filename)
    MODEL = joblib.load(pattern_path)
    CENTROIDS = MODEL["centroids"]
    STDS = MODEL["stds"]
    feats = compute_features(df)          # même fonction qu'à l'entraînement
    if feats.empty:
        df["candle_type"] = 0
        return df

    X   = feats.values.astype(np.float32)
    Xw  = X / STDS                       # <-- même whitening
    labels, _ = vq(Xw, CENTROIDS)        # simple quantisation, ultra-rapide
    df["candle_type"] = labels
    return df

def bucket_variation(row, thresholds=(-0.01, 0.01)):
    try:
        v = float(row["variation_pct"])
        if v < thresholds[0]:
            return -1
        elif v > thresholds[1]:
            return 1
        return 0
    except Exception as e:
        print("Erreur variation:", e)
        return 0

# Application


def detect_known_patterns(df, known_patterns, max_len=3, max_results=5, min_gap_minutes=2):
    # Calcul de la variation en %
    df["variation_pct"] = ((df['close'] - df['open']) / df['open']).astype(float).fillna(0)
    df["variation_bucket"] = df.apply(lambda row: bucket_variation(row), axis=1)
    sequence_tuples = list(zip(df["candle_type"], df["variation_bucket"]))
    timestamps = pd.to_datetime(df['timestamp_utc']).tolist()
    matches = []

    for i in range(len(sequence_tuples)):
        for l in range(1, max_len + 1):
            if i + l > len(sequence_tuples):
                continue
            seq = tuple(sequence_tuples[i:i + l])
            if seq in known_patterns:
                match = {
                    "sequence": seq,
                    "start_timestamp": timestamps[i].isoformat(),
                    "end_timestamp": timestamps[i + l - 1].isoformat(),
                    "bias": known_patterns[seq]["bias"],
                    "bullish_ratio": known_patterns[seq]["bullish_ratio"],
                    "bearish_ratio": known_patterns[seq]["bearish_ratio"],
                    "neutral_ratio": known_patterns[seq].get("neutral_ratio",
                                                             1.0 - known_patterns[seq]["bullish_ratio"] -
                                                             known_patterns[seq]["bearish_ratio"]),
                    "direction": (
                        "bullish" if known_patterns[seq]["bias"] > 0.05 else
                        "bearish" if known_patterns[seq]["bias"] < -0.05 else "neutral"
                    )
                }

                matches.append(match)

    matches = sorted(matches, key=lambda x: abs(x["bias"]), reverse=True)

    filtered = []
    seen_sequences = set()
    last_end_time = None

    for match in matches:
        key = (match["sequence"], match["direction"])

        if key in seen_sequences:
            continue

        if last_end_time:
            start_time = pd.to_datetime(match["start_timestamp"])
            if (start_time - last_end_time).total_seconds() / 60 < min_gap_minutes:
                continue

        filtered.append(match)
        seen_sequences.add(key)
        last_end_time = pd.to_datetime(match["end_timestamp"])

        if len(filtered) >= max_results:
            break

    return filtered

# === Endpoint principal ===
@app.get("/load-data-patterns")
def load_data_pattern(
    symbol: str = Query(...),
    start_date: str = Query(..., description="YYYY-MM-DDTHH:MM"),
    end_date: str = Query(..., description="YYYY-MM-DDTHH:MM")
):
    start_utc = parse_min(start_date)
    end_utc = parse_min(end_date)

    df = load_crypto_data_custom_range(symbol=symbol, start_date=start_utc, end_date=end_utc)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    df = assign_candle_types(df)

    patterns = detect_known_patterns(df, load_patterns())

    if len(patterns) > 0:
        last = patterns[-1]
        direction = last['direction']
        prob = abs(last['bias'])
        short_term_forecast = {
            "direction": direction,
            "probability": min(1.0, max(0.5, 0.5 + prob)),
            "bias": round(last['bias'], 3)
        }
    else:
        short_term_forecast = None

    return {
        "symbol": symbol,
        "start_date": start_utc.isoformat(),
        "end_date": end_utc.isoformat(),
        "patterns_detected": patterns,
        "short_term_forecast": short_term_forecast
    }

@app.get("load-data-patterns-classic")
def patterns_classic(
    symbol: str = Query(..., examples={"BTCUSDT": { "summary": "Bitcoin/USDT" }}),
    start_date: str = Query(..., description="YYYY-MM-DDTHH:MM (local)"),
    end_date: str   = Query(..., description="YYYY-MM-DDTHH:MM (local)"),
    atr_min_pct: float = Query(0.05, description="ATR filter in % (volatility guard)")
):
    """
    Renvoie la liste des patterns chandeliers « classiques » détectés
    entre start_date et end_date sur le symbole donné.
    """
    # 1) Conversion des dates en UTC (même logique que tes autres endpoints)
    start_utc = parse_min(start_date)
    end_utc = parse_min(end_date)

    # 2) Récupération des bougies (ta fonction maison)
    df = load_crypto_data_custom_range(
        symbol=symbol,
        start_date=start_utc,
        end_date=end_utc
    ).sort_values("timestamp_utc").reset_index(drop=True)

    if df.empty:
        return {
            "symbol": symbol,
            "start_date": start_utc.isoformat(),
            "end_date": end_utc.isoformat(),
            "patterns_detected": []
        }

    # 3) Détection des motifs
    patt_df: pd.DataFrame = detect_classic_patterns(df, atr_min_pct=atr_min_pct)

    # 4) Renvoi JSON — chaque pattern est un dict
    return {
        "symbol": symbol,
        "start_date": start_utc.isoformat(),
        "end_date": end_utc.isoformat(),
        "patterns_detected": patt_df.to_dict(orient="records")
    }


#uvicorn PA_ML.crypto_forecast_ml.predictor.serve_api:app --port 8000 --reload

