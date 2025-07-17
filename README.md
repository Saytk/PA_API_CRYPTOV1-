# Crypto Trading Bot - Améliorations

Ce projet a été amélioré pour optimiser les performances du bot de trading crypto et intégrer différentes fonctionnalités dans une API unifiée. Voici les principales modifications apportées et leurs bénéfices attendus.

## API Unifiée

Une nouvelle API unifiée a été créée pour intégrer les fonctionnalités des modules PA_ML et crypto_signals, ainsi que pour ajouter des fonctionnalités de gestion de portefeuille. Cette API fournit une interface complète pour l'analyse des cryptomonnaies, la prédiction, la détection de patterns et la gestion de portefeuille.

### Utilisation de l'API unifiée

Pour démarrer l'API unifiée, exécutez :

```bash
python -m crypto_api.run_api
```

L'API sera disponible à l'adresse `http://localhost:8000`.

Pour plus d'informations sur l'API unifiée, consultez la [documentation de l'API](crypto_api/README.md).

## Améliorations principales

### 1. Feature Engineering avancé
- **Ajout de nombreux indicateurs techniques** : RSI, Stochastique, ADX, CCI, MFI, etc.
- **Indicateurs de volume améliorés** : OBV, VWAP, ratios de volume
- **Caractéristiques de microstructure** : taille du corps des bougies, ombres, etc.
- **Patterns de prix** : supports/résistances, position relative aux moyennes mobiles

### 2. Entraînement du modèle optimisé
- **Validation croisée temporelle** : meilleure évaluation de la performance sur des données temporelles
- **Hyperparamètres étendus** : recherche plus large et plus profonde des meilleurs paramètres
- **Régularisation améliorée** : L1 et L2 pour éviter le surapprentissage
- **Early stopping avec patience accrue** : permet une convergence plus stable

### 3. Backtest complet et analyse
- **Métriques de performance détaillées** : précision, rappel, F1-score, AUC ROC
- **Analyse des trades** : taux de réussite, rendement moyen, rendement cumulatif
- **Visualisations** : distribution des probabilités, courbe ROC, matrice de confusion
- **Comparaison entre actifs** : permet d'identifier les actifs les plus performants

### 4. Simplification du code
- **Architecture orientée objet** : code plus modulaire et maintenable
- **Meilleure gestion des erreurs** : logging amélioré et gestion des cas limites
- **Documentation complète** : docstrings et commentaires explicatifs

## Comment utiliser le bot amélioré

### Entraînement des modèles
```bash
python -m crypto_signals.src.train_lgbm
```
Cette commande entraînera des modèles optimisés pour tous les actifs configurés dans `config.yaml`.

### Exécution des backtests
```bash
python -m crypto_signals.src.backtest
```
Cette commande exécutera des backtests complets sur tous les actifs et générera des visualisations dans le dossier `results/`.

### Prédictions en temps réel
```bash
python -m crypto_signals.src.predict
```
Cette commande générera des prédictions pour le dernier point de données disponible.

## Résultats attendus

Les améliorations apportées devraient conduire à :

1. **Meilleure précision des prédictions** : grâce aux features avancées et à l'optimisation du modèle
2. **Réduction du surapprentissage** : grâce à la validation croisée et à la régularisation
3. **Signaux de trading plus fiables** : grâce au seuil de probabilité ajusté (0.6 au lieu de 0.5)
4. **Meilleure compréhension des performances** : grâce aux métriques et visualisations détaillées

## Pistes d'amélioration futures

1. **Ensemble de modèles** : combiner plusieurs algorithmes pour des prédictions plus robustes
2. **Features basées sur l'orderbook** : intégrer des données de profondeur de marché
3. **Optimisation du seuil de décision** : trouver le meilleur compromis précision/rappel
4. **Gestion du risque avancée** : sizing des positions basé sur la confiance du modèle
5. **Analyse de sentiment** : intégrer des données de sentiment du marché depuis les réseaux sociaux
