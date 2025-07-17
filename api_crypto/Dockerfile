# ✅ Choisir une version stable de Python (3.10 pour compatibilité ML)
FROM python:3.10-slim

# ✅ Empêcher Python d'écrire des fichiers .pyc (inutile en prod)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# ✅ Installer les dépendances système pour SciPy / XGBoost / LightGBM / TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ✅ Créer un dossier pour l'application
WORKDIR /app

# ✅ Copier uniquement requirements.txt d'abord (pour profiter du cache Docker)
COPY requirements.txt .

# ✅ Mettre à jour pip, setuptools et wheel AVANT d'installer les libs
RUN pip install --upgrade pip setuptools wheel

# ✅ Installer les dépendances Python (en wheels si possible)
RUN pip install -r requirements.txt

# ✅ Copier le reste du projet
COPY . .

# ✅ Exposer le port (Render en fournit un via $PORT)
EXPOSE 8000

# ✅ Commande par défaut : lancer Uvicorn
CMD ["uvicorn", "crypto_api.api:app", "--host", "0.0.0.0", "--port", "8000"]
