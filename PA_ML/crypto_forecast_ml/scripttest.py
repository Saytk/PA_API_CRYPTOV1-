# crypto_forecast_ml/scripttest.py

import logging
import argparse
import os
from training.train_model import train_direction_model_with_timerange

# ✅ Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)

if __name__ == "__main__":
    # Parsing des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Script d'entraînement du modèle de prédiction crypto")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbole de la paire (ex: BTCUSDT)")
    parser.add_argument("--hours", type=int, help="Nombre d'heures de données à utiliser pour l'entraînement")
    parser.add_argument("--days", type=int, default=7, help="Nombre de jours de données si hours n'est pas spécifié")
    parser.add_argument("--all-data", action="store_true", help="Utiliser toutes les données disponibles sans contrainte de temps")
    parser.add_argument("--output", type=str, default="models/xgb_direction.json", help="Chemin de sortie du modèle")

    args = parser.parse_args()

    logging.info("🚀 Démarrage du script d'entraînement")
    logging.info(f"Paramètres: symbol={args.symbol}, hours={args.hours}, days={args.days}, all_data={args.all_data}, output={args.output}")

    # Utilisation de la nouvelle fonction qui gère tout le processus
    df = train_direction_model_with_timerange(
        symbol=args.symbol,
        hours=args.hours,
        days=args.days,
        all_data=args.all_data,
        output_path=args.output
    )

    # Obtenir le chemin absolu pour le logging
    if os.path.isabs(args.output):
        abs_path = args.output
    else:
        # Si le chemin est relatif, le convertir en absolu
        current_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(current_dir, args.output)

    logging.info(f"✅ Script terminé avec succès. Modèle sauvegardé dans {args.output}")
    logging.info(f"✅ Chemin absolu du modèle : {abs_path}")
    logging.info(f"📊 Nombre total de lignes traitées: {len(df)}")
