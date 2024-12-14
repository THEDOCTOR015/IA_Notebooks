#!/bin/bash

# Lancer Flask pour saisir la clé API
echo "Lancement du serveur Flask pour la clé API..."
python web/web.py &

# Démarrer le backend ML une fois la clé API fournie
echo "En attente de la clé API pour démarrer le backend ML..."
python start_ml_backend.py
