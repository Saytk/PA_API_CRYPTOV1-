"""
Code de diagnostic pour identifier les problèmes dans vos données
Exécutez ce code AVANT de continuer l'entraînement
"""

import pandas as pd
import numpy as np
from crypto_signals.src.train_lgbm import prepare


def diagnostic_complet(symbol="BTCUSDT"):
    """Diagnostic complet des données d'entraînement."""
    print(f"=== DIAGNOSTIC POUR {symbol} ===\n")

    # 1. Charger les données
    df = prepare(symbol)
    print(f"📊 Nombre total de lignes: {len(df)}")

    # 2. Analyser la distribution du target
    target_dist = df['target'].value_counts()
    print(f"\n🎯 Distribution du target:")
    print(f"  - Classe 0 (pas de hit): {target_dist[0]} ({target_dist[0] / len(df) * 100:.1f}%)")
    print(f"  - Classe 1 (hit): {target_dist[1]} ({target_dist[1] / len(df) * 100:.1f}%)")

    # 3. Vérifier la fuite de données
    print(f"\n🔍 Vérification de la fuite de données:")

    # Calculer manuellement le target sans fuite
    margin = 0.0016
    horizon = 5

    # Méthode actuelle (avec fuite potentielle)
    current_target = df['target'].copy()

    # Méthode corrigée
    corrected_targets = []
    for i in range(len(df)):
        if i + horizon < len(df):
            future_high = df["high"].iloc[i + 1:i + 1 + horizon].max()
            hit = future_high >= df["close"].iloc[i] * (1 + margin)
            corrected_targets.append(int(hit))
        else:
            corrected_targets.append(np.nan)

    df['target_corrected'] = corrected_targets
    df_clean = df.dropna()

    # Comparer les distributions
    if len(df_clean) > 0:
        corrected_dist = df_clean['target_corrected'].value_counts()
        print(f"  - Target corrigé - Classe 0: {corrected_dist[0]} ({corrected_dist[0] / len(df_clean) * 100:.1f}%)")
        print(f"  - Target corrigé - Classe 1: {corrected_dist[1]} ({corrected_dist[1] / len(df_clean) * 100:.1f}%)")

        # Différence entre les deux méthodes
        diff = (df_clean['target'] != df_clean['target_corrected']).sum()
        print(f"  - Différences entre méthodes: {diff} lignes ({diff / len(df_clean) * 100:.1f}%)")

    # 4. Analyser les features
    print(f"\n📈 Analyse des features:")
    for col in ['close', 'ret_1', 'ret_5', 'vol_30']:
        if col in df.columns:
            print(f"  - {col}: min={df[col].min():.6f}, max={df[col].max():.6f}, std={df[col].std():.6f}")

    # 5. Vérifier les valeurs manquantes
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n❌ Valeurs manquantes détectées:")
        print(missing[missing > 0])
    else:
        print(f"\n✅ Pas de valeurs manquantes")

    # 6. Analyser la corrélation avec le futur (signe de fuite)
    print(f"\n🔮 Corrélation avec les prix futurs (signe de fuite):")
    if 'ret_1' in df.columns:
        future_ret = df['ret_1'].shift(-1)
        correlation = df['ret_1'].corr(future_ret)
        print(f"  - Corrélation ret_1 avec ret_1 futur: {correlation:.4f}")
        if abs(correlation) > 0.05:
            print(f"  ⚠️  Corrélation élevée détectée - possible fuite!")

    # 7. Tester la prédictibilité baseline
    print(f"\n🎲 Test de prédictibilité baseline:")

    # Modèle naive : prédire toujours la classe majoritaire
    majority_class = df['target'].mode()[0]
    baseline_accuracy = (df['target'] == majority_class).mean()
    print(f"  - Accuracy baseline (toujours {majority_class}): {baseline_accuracy:.4f}")

    # Si accuracy > 0.95, c'est suspect
    if baseline_accuracy > 0.95:
        print(f"  ⚠️  Accuracy baseline très élevée - données déséquilibrées!")

    # 8. Vérifier les outliers
    print(f"\n📊 Analyse des outliers:")
    close_q99 = df['close'].quantile(0.99)
    close_q01 = df['close'].quantile(0.01)
    outliers = ((df['close'] > close_q99) | (df['close'] < close_q01)).sum()
    print(f"  - Outliers de prix (1% extrêmes): {outliers} ({outliers / len(df) * 100:.1f}%)")

    return df


def test_validation_split():
    """Test de la validation croisée pour détecter les fuites."""
    print(f"\n=== TEST DE VALIDATION CROISÉE ===")

    # Simuler des données avec fuite connue
    np.random.seed(42)
    n_samples = 1000

    # Créer des données avec fuite intentionnelle
    X = np.random.randn(n_samples, 5)
    y = np.random.randint(0, 2, n_samples)

    # Introduire une fuite : les features dépendent du futur
    for i in range(n_samples - 10):
        if y[i + 5] == 1:  # Si le target futur est 1
            X[i, 0] += 2  # Augmenter une feature

    # Tester avec validation croisée classique vs temporelle
    from crypto_signals.src.utils.custom_metrics import cross_val_score, KFold, LogisticRegression

    # Validation classique (shuffle)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_shuffle = cross_val_score(LogisticRegression(), X, y, cv=kfold)

    # Validation temporelle (sans shuffle)
    kfold_temporal = KFold(n_splits=5, shuffle=False)
    scores_temporal = cross_val_score(LogisticRegression(), X, y, cv=kfold_temporal)

    print(f"  - Score avec shuffle: {scores_shuffle.mean():.4f} (±{scores_shuffle.std():.4f})")
    print(f"  - Score temporel: {scores_temporal.mean():.4f} (±{scores_temporal.std():.4f})")

    if scores_shuffle.mean() > scores_temporal.mean() + 0.1:
        print(f"  ⚠️  Différence significative détectée - possible fuite de données!")

    return scores_shuffle, scores_temporal


# Exécuter le diagnostic
if __name__ == "__main__":
    # Diagnostic principal
    df = diagnostic_complet("BTCUSDT")

    # Test de validation
    test_validation_split()

    print(f"\n" + "=" * 60)
    print("RECOMMANDATIONS:")
    print("1. Si différences dans les targets > 5%, corrigez la fuite")
    print("2. Si accuracy baseline > 95%, rééquilibrez les données")
    print("3. Si corrélation future > 0.05, vérifiez vos features")
    print("4. Si scores shuffle >> temporel, implémentez validation temporelle")
    print("=" * 60)
