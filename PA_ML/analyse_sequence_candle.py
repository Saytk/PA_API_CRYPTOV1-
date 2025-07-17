import pandas as pd
import numpy as np
from collections import Counter

# === Paramètres ===
CSV_PATH = "BTCUSDT_labeled_candles.csv"
MAX_SEQ_LEN = 3
TARGET_HORIZON = 5           # minutes après la fin de la séquence
TOLERANCE_PCT = 0.1          # seuil neutre : ±0.1%
VARIATION_THRESHOLDS = (-0.001, 0.001)  # seuils bucket: -0.1%, 0.1%

# === Charger les données ===
df = pd.read_csv(CSV_PATH)
df = df.sort_values("timestamp_utc").reset_index(drop=True)

# === Calculer variation en % et bucket associé ===
df["variation_pct"] = (df["close"] - df["open"]) / df["open"]

def bucket_variation(pct):
    if pct < VARIATION_THRESHOLDS[0]:
        return -1
    elif pct > VARIATION_THRESHOLDS[1]:
        return 1
    return 0

df["variation_bucket"] = df["variation_pct"].apply(bucket_variation)

# === Générer les séquences de 1 à MAX_SEQ_LEN ===
records = []
close_prices = df['close'].tolist()
combined_seq = list(zip(df['candle_type'], df['variation_bucket']))

for i in range(len(df) - TARGET_HORIZON):
    for seq_len in range(1, MAX_SEQ_LEN + 1):
        end = i + seq_len
        if end + TARGET_HORIZON >= len(df):
            continue

        seq = tuple(combined_seq[i:end])
        now_price = close_prices[end - 1]
        fut_price = close_prices[end - 1 + TARGET_HORIZON]
        change_pct = (fut_price - now_price) / now_price

        if change_pct > TOLERANCE_PCT / 100:
            label = 1
        elif change_pct < -TOLERANCE_PCT / 100:
            label = -1
        else:
            label = 0

        records.append((seq, label))

# === Statistiques : séquence → distribution des labels ===
stats = {}
for seq, label in records:
    if seq not in stats:
        stats[seq] = Counter()
    stats[seq][label] += 1

# === Construire le DataFrame des résultats ===
rows = []
for seq, counter in stats.items():
    total = sum(counter.values())
    bullish = counter[1]
    bearish = counter[-1]
    neutral = counter[0]
    rows.append({
        'sequence': seq,
        'total': total,
        'bullish': bullish,
        'bearish': bearish,
        'neutral': neutral,
        'bullish_ratio': bullish / total,
        'bearish_ratio': bearish / total,
        'neutral_ratio': neutral / total,
        'bias': (bullish - bearish) / total
    })

result_df = pd.DataFrame(rows).sort_values(by='total', ascending=False)

# === Afficher les top séquences ===
pd.set_option('display.max_rows', 50)
print("\nTop séquences par fréquence :")
print(result_df.head(20)[['sequence', 'total', 'bullish_ratio', 'bearish_ratio', 'bias']])

print("\nTop séquences haussières :")
print(result_df.sort_values(by='bias', ascending=False).head(20)[['sequence', 'total', 'bullish_ratio', 'bias']])

print("\nTop séquences baissières :")
print(result_df.sort_values(by='bias').head(20)[['sequence', 'total', 'bearish_ratio', 'bias']])

# === Filtrer les plus puissantes ===
MIN_OCCURRENCES = 30
MIN_BIAS = 0.1

filtered_df = result_df[
    (result_df['total'] >= MIN_OCCURRENCES) &
    (result_df['bias'].abs() >= MIN_BIAS)
].copy()

filtered_df = filtered_df.sort_values(by='bias', ascending=False)

print(f"\n🧠 Séquences directionnelles solides (≥{MIN_OCCURRENCES} cas, |bias| ≥ {MIN_BIAS}):")
print(filtered_df[['sequence', 'total', 'bullish_ratio', 'bearish_ratio', 'bias']].head(30))

filtered_df.to_csv("patterns_significatifs.csv", index=False)
print("📦 Exporté vers patterns_significatifs.csv")
