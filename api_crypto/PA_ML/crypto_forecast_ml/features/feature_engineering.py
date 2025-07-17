import pandas as pd
import numpy as np
import ta


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute tous les indicateurs techniques nécessaires pour le modèle.
    Assure que les colonnes sont bien alignées avec celles utilisées à l'entraînement.
    """
    df = df.copy()

    # ✅ Simple Moving Averages (SMA)
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_10"] = df["close"].rolling(window=10).mean()

    # ✅ Exponential Moving Averages (EMA)
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()

    # ✅ RSI
    df["rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    # ✅ MACD
    macd = ta.trend.MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # ✅ Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    # ✅ ATR (Average True Range)
    atr = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr_14"] = atr.average_true_range()

    # ✅ Nettoyage : supprime les lignes avec NaN dues aux indicateurs
    df = df.dropna().reset_index(drop=True)

    return df
