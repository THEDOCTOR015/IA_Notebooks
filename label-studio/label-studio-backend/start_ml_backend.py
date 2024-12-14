import os
import time
import subprocess

def wait_for_api_key():
    print("En attente de la clé API...")
    api_key_path = '/app/api_key.txt'
    while not os.path.exists(api_key_path):  # Attendre que le fichier soit créé
        time.sleep(2)
    with open(api_key_path, 'r') as file:
        api_key = file.read().strip()
    print(f"Clé API reçue : {api_key}")
    os.environ['LABEL_STUDIO_API_KEY'] = api_key  # Définir la clé API dans l'environnement

def start_ml_backend():
    print("Démarrage du backend ML...")
    subprocess.run(["python", "model.py"])

if __name__ == '__main__':
    wait_for_api_key()
    start_ml_backend()
