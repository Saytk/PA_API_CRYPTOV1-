import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path.cwd()))

# Import the train function from train_lgbm.py
from crypto_signals.src.train_lgbm import train

# Run the train function with the default symbol (BTCUSDT)
if __name__ == "__main__":
    train("BTCUSDT")