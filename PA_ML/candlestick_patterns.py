# candlestick_patterns.py
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. Choix de la librairie : TA-Lib → fallback pandas_ta -----------
# ------------------------------------------------------------------
try:
    import talib as ta
    LIB = "talib"
except ImportError:
    import pandas_ta as ta  # pip install pandas_ta
    LIB = "pandas_ta"

# ------------------------------------------------------------------
# 2. Table des motifs retenus --------------------------------------
# ------------------------------------------------------------------
PATTERNS = {
    "CDLHAMMER": {
        "name": "Hammer",
        "summary": "Potential bullish reversal.",
        "detail": "Small body, long lower wick – buyers stepping in.",
        "dir_from_sign": True       # +100 → bullish, -100 → bearish (rare)
    },
    "CDLENGULFING": {
        "name": "Engulfing",
        "summary": "Strong reversal signal.",
        "detail": "Second candle’s body engulfs the previous one.",
        "dir_from_sign": True
    },
    "CDLSHOOTINGSTAR": {
        "name": "Shooting Star",
        "summary": "Potential bearish reversal.",
        "detail": "Small body near low, long upper shadow – supply overtakes demand.",
        "direction": "bearish"
    },
    "CDLHANGINGMAN": {
        "name": "Hanging Man",
        "summary": "Caution: bearish warning in an uptrend.",
        "detail": "Lower wick > 2× body after rise.",
        "direction": "bearish"
    },
    "CDLDOJI": {
        "name": "Doji",
        "summary": "Indecision – watch breakout.",
        "detail": "Open ≈ Close, long shadows indicate tug-of-war.",
        "direction": "neutral"
    },
    "CDLMORNINGSTAR": {
        "name": "Morning Star",
        "summary": "Three-candle bullish reversal.",
        "detail": "Gap down, small indecision, strong green close.",
        "direction": "bullish"
    },
    "CDLEVENINGSTAR": {
        "name": "Evening Star",
        "summary": "Three-candle bearish reversal.",
        "detail": "Gap up, indecision, strong red close.",
        "direction": "bearish"
    },
    "CDLPIERCING": {
        "name": "Piercing Line",
        "summary": "Bullish reversal after decline.",
        "detail": "Second candle opens lower, closes > 50 % of previous body.",
        "direction": "bullish"
    },
    "CDLDARKCLOUDCOVER": {
        "name": "Dark Cloud Cover",
        "summary": "Bearish reversal after rise.",
        "detail": "Opens higher, closes < 50 % of previous body.",
        "direction": "bearish"
    }
}

# ------------------------------------------------------------------
# 3. Détection principale ------------------------------------------
# ------------------------------------------------------------------
def detect_classic_patterns(df: pd.DataFrame,
                            atr_period: int = 14,
                            atr_min_pct: float = 0.1) -> pd.DataFrame:
    """
    df doit contenir : ['timestamp_utc','open','high','low','close']
    atr_min_pct    : seuil volatilité (ex. 0.1 = 0.1 %)
    Retourne un DataFrame des détections.
    """
    # --- préparer les colonnes ohcl en ndarray (TA-Lib aime bien) --
    o, h, l, c = [df[x].values for x in ['open','high','low','close']]

    # --- ATR pour filtrage volatilité --------------------------------
    if LIB == "talib":
        atr = ta.ATR(o, h, l, timeperiod=atr_period)
    else:  # pandas_ta
        atr = ta.atr(high=df['high'], low=df['low'], close=df['close'],
                     length=atr_period).values
    # Variation relative
    atr_pct = atr / df['close'].values * 100

    detections = []

    for code, meta in PATTERNS.items():
        # --- appel fonction base ------------------------------------
        if LIB == "talib":
            values = getattr(ta, code)(o, h, l, c)
        else:
            values = ta.cdl_pattern(df, name=code[3:].lower()).values

        idxs = np.where(values != 0)[0]   # positions non-nulles
        for i in idxs:
            # Volatilité suffisante ?
            if atr_pct[i] < atr_min_pct:
                continue

            sign = int(np.sign(values[i]))
            # Direction déduite du signe si demandé
            direction = meta.get("direction")
            if meta.get("dir_from_sign"):
                direction = "bullish" if sign > 0 else "bearish"

            detections.append({
                "timestamp": df.at[i, 'timestamp_utc'],
                "pattern_code": code,
                "pattern": meta["name"],
                "direction": direction,
                "summary": meta["summary"],
                "detail": meta["detail"]
            })

    return pd.DataFrame(detections)

# ------------------------------------------------------------------
# 4. Exemple d'utilisation -----------------------------------------
# ------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("BTCUSDT_1m.csv")            # tes données OHLCV
    out = detect_classic_patterns(df, atr_min_pct=0.05)
    print(out.head(20))
    # ⇒ tu peux ensuite exporter ou intégrer à ton API
