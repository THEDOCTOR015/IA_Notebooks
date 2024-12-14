from flask import Flask, request, render_template
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        api_key = request.form.get('api_key')
        if api_key:
            # Stockez la clé API dans un fichier
            with open('/app/api_key.txt', 'w') as file:
                file.write(api_key)
            return f'Clé API enregistrée avec succès : {api_key}'
        else:
            return render_template('index.html', error="Veuillez entrer une clé API valide.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
