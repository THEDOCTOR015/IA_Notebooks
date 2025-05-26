FROM tensorflow/tensorflow:latest-gpu-jupyter
WORKDIR /tf
# Copier les dépendances
COPY requirements.txt .

# Installer les packages Python globalement (optionnel : --no-cache-dir)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Génère le mot de passe Jupyter (argon2) à partir de la variable ARG
ARG JUPYTER_PLAIN_PASSWORD=admin

# Installer les dépendances système nécessaires pour CV2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Installer Jupyter Server et configurer le mot de passe
RUN pip install jupyter_server && \
    python3 -c "from jupyter_server.auth import passwd; print(passwd('${JUPYTER_PLAIN_PASSWORD}'))" > /tmp/jupyter_pass

# Configure le mot de passe dans la config Jupyter
RUN mkdir -p /root/.jupyter && \
    python3 -c "pwd = open('/tmp/jupyter_pass').read().strip(); open('/root/.jupyter/jupyter_server_config.py', 'w').write(f'c.ServerApp.password = \"{pwd}\"\n')"


# Exposer le port Jupyter
EXPOSE 8888

# Lancer le serveur Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]