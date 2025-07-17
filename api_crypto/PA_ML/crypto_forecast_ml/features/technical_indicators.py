# features/technical_indicators.py

import pandas as pd
import ta


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les indicateurs techniques standard à un DataFrame OHLCV.

    Args:
        df (pd.DataFrame): OHLCV avec colonnes ['open', 'high', 'low', 'close', 'volume']

    Returns:
        pd.DataFrame: DataFrame enrichi avec les colonnes techniques
    """
    df = df.copy()

    # Initialisation des indicateurs via ta (technical analysis)
    df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['ema_5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

    macd = ta.trend.macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['macd_diff'] = ta.trend.macd_diff(df['close'])

    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()

    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    return df
