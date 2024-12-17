from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.api import init_app
import os
import glob
from PIL import Image
import numpy as np
import time
import requests
from io import BytesIO
from tensorflow.keras.models import load_model
import tensorflow as tf
import importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_model_and_config(model_base_folder):
    """
    Charge un modèle Keras et ses paramètres de configuration associés.

    Arguments :
        model_base_folder (str) : Le dossier contenant les modèles "yolo_MM_DD" avec leurs configs.

    Retourne :
        model (tensorflow.keras.Model) : Le modèle Keras chargé.
        config (module) : Un objet contenant les paramètres de configuration.
    """
    # Rechercher les dossiers "yolo_MM_DD" dans le dossier de base
    model_folders = glob.glob(os.path.join(model_base_folder, "yolo_*"))
    if not model_folders:
        raise FileNotFoundError("Aucun dossier de modèle 'yolo_MM_DD' trouvé.")
    
    # Trier pour choisir le dernier modèle en date
    latest_model_folder = max(model_folders, key=os.path.getctime)
    print(f"Modèle sélectionné : {latest_model_folder}", flush=True)
    
    # Charger le fichier de configuration (config.py) en tant que module dynamique
    config_path = os.path.join(latest_model_folder, "config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Fichier de configuration non trouvé : {config_path}")
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Charger le modèle Keras
    model_path = os.path.join(latest_model_folder, "yolo_*.h5")
    model_files = glob.glob(model_path)
    if not model_files:
        raise FileNotFoundError(f"Aucun fichier modèle '.h5' trouvé dans : {latest_model_folder}")
    
    model_file = model_files[0]
    print(f"Chargement du modèle : {model_file}", flush=True)
    model = load_model(model_file)
    
    return model, config


class YOLO(LabelStudioMLBase):

    @tf.function
    def optimized_predict(model, image):
        return model(image)

    def __init__(self, **kwargs):
        super(YOLO, self).__init__(**kwargs)

    def _download_and_preprocess_image(self,image_url):
        
        # Télécharger et prétraiter l'image
        headers = {
        'Authorization': f'Token {API_KEY}'
        }
        res = requests.get(image_url, headers=headers)
        res.raise_for_status()  # Vérifiez les erreurs HTTP
        img = Image.open(BytesIO(res.content))
        img = img.convert("RGB")  # S'assurer que l'image est RGB
        img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))  # Redimensionner
        image_array = np.array(img) / 255.0  # Normalisation entre 0 et 1
        return np.expand_dims(image_array, axis=0) # Ajouter une dimension batch

    def _post_process_predictions(self, pred):
        def sigmoid(x):
            x=np.clip(x, -50, 50)
            return 1/(1+np.exp(-x))
        boxes = []
        pred = pred[0] # Enlever le batch
        pred_boxes=pred[..., 0:4]
        pred_conf=tf.sigmoid(pred[..., 4])
        for i in range(MODEL_CELLULES[0]):
            for j in range(MODEL_CELLULES[1]):
                k_max = np.argmax(pred_conf[i, j])
                box = pred_boxes[i, j, k_max]
                if pred_conf[i,j,k_max] > THRESHOLD_CONFIDENCE :
                    # Conversion de la box prédite en coordonnées pixels (on prend en compte le resize de l'image)
                    pixel_x_center = (j + MODEL_GRID_SENSIBILITY_COEF*sigmoid(box[0]) - (MODEL_GRID_SENSIBILITY_COEF-1)/2) * MODEL_CELLULES_SIZE[1]
                    pixel_y_center = (i + MODEL_GRID_SENSIBILITY_COEF*sigmoid(box[1]) - (MODEL_GRID_SENSIBILITY_COEF-1)/2) * MODEL_CELLULES_SIZE[0]
                    pixel_w_box = (MODEL_SIGMOID_MULTIPLIER*sigmoid(box[2]) + MODEL_SIGMOID_ADDER ) * MODEL_ANCHOR_BOXES[k_max][0] * MODEL_CELLULES_SIZE[1]
                    pixel_h_box = (MODEL_SIGMOID_MULTIPLIER*sigmoid(box[3]) + MODEL_SIGMOID_ADDER ) * MODEL_ANCHOR_BOXES[k_max][1] * MODEL_CELLULES_SIZE[0]

                    # Conversion pour le coin en haut à gauche (pour label-studio)
                    topleft_pixel_x_center = pixel_x_center - pixel_w_box/2
                    topleft_pixel_y_center = pixel_y_center - pixel_h_box/2
                    topleft_pixel_w_box = pixel_w_box
                    topleft_pixel_h_box = pixel_h_box

                    # Conversion de la box en coordonnées % 
                    p100_x = (topleft_pixel_x_center / IMAGE_SIZE[1]) * 100
                    p100_y = (topleft_pixel_y_center / IMAGE_SIZE[0]) * 100
                    p100_w = (topleft_pixel_w_box / IMAGE_SIZE[1]) * 100
                    p100_h = (topleft_pixel_h_box / IMAGE_SIZE[0]) * 100
                    box = {
                        'x' : p100_x,
                        'y' : p100_y,
                        'width' : p100_w,
                        'height' : p100_h
                    }
                
                    boxes.append(box)
        return boxes
    
    def predict(self, tasks, **kwargs):
        results = []
        for task in tasks:
            result_task = []
            image_url = task['data']['image']
            image_url = LABEL_STUDIO_BASEURL + image_url
            image = self._download_and_preprocess_image(image_url) # w/ batch
            start_time = time.perf_counter()  # Démarre le chronomètre
            pred = YOLO.optimized_predict(YOLO.model, image)
            end_time = time.perf_counter()  # Arrête le chronomètre
            
            inference_time_ms = (end_time - start_time) * 1000  # Temps en millisecondes
            print(f"Temps d'inférence : {inference_time_ms:.2f} ms", flush=True)
            # post process
            boxes = self._post_process_predictions(pred)

            # json
            for box in boxes :
                result_box = {}
                result_box['from_name'] = 'label'
                result_box['to_name'] = 'image'
                result_box['type'] = 'rectanglelabels'
                result_box['value'] = {
                                "x": float(box["x"]),               # Coordonnée x en pourcentage (en haut à gauche)
                                "y": float(box["y"]),               # Coordonnée y en pourcentage ( en haut à gauche)
                                "width": float(box["width"]),       # Largeur en pourcentage
                                "height": float(box["height"]),     # Hauteur en pourcentage
                                'rectanglelabels': ['Object']
                    }
                result_task.append(result_box)
        
        results.append({
        'result': result_task
        })

        return results

if __name__ == "__main__":
    UNLABELED_IMAGES_FOLDER = "/data/unlabeled/"  
    OUTPUT_LABELS_FOLDER = "/data/labels/"
    MODEL_FOLDER = "/app/models/"
    LABEL_STUDIO_BASEURL = 'http://label-studio:8080'
    # Setup model
    model, config = load_model_and_config(MODEL_FOLDER)
    # Paramètres du modèle
    MODEL_ANCHOR_BOXES = config.MODEL_ANCHOR_BOXES
    MODEL_CELLULES = config.MODEL_CELLULES
    IMAGE_SIZE = config.IMAGE_SIZE
    MODEL_CELLULES_SIZE = config.MODEL_CELLULES_SIZE
    THRESHOLD_CONFIDENCE = config.THRESHOLD_CONFIDENCE
    MODEL_MINIMAL_IOU = config.MODEL_MINIMAL_IOU
    MODEL_GRID_SENSIBILITY_COEF = config.MODEL_GRID_SENSIBILITY_COEF
    MODEL_SIGMOID_MULTIPLIER = config.MODEL_SIGMOID_MULTIPLIER
    MODEL_SIGMOID_ADDER = config.MODEL_SIGMOID_ADDER
    YOLO.model = model
    API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
    print(f'ML Backend | Clé API : {API_KEY}', flush=True)
    app = init_app(YOLO)
    app.run(host="0.0.0.0", port=9090, debug=True)