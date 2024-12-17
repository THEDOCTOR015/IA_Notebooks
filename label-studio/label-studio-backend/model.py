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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class YOLO(LabelStudioMLBase):
    model = None  # Attribut de classe, partagé entre toutes les instances

    @tf.function
    def optimized_predict(model, image):
        return model(image)

    def __init__(self, **kwargs):
        super(YOLO, self).__init__(**kwargs)
        
        if YOLO.model is None:  # Vérifie si le modèle est déjà chargé
            print(f'Chargement du modèle YOLO en cours', flush=True)
            # Localiser et charger le fichier du modèle YOLO
            model_files = glob.glob(os.path.join(MODEL_FOLDER, "yolo_*"))
            if not model_files:
                raise FileNotFoundError("Aucun modèle YOLO trouvé dans le dossier spécifié.")
            
            # Prendre le modèle avec la date la plus récente
            latest_model = max(model_files, key=os.path.getctime)
            print(f"Chargement du modèle : {latest_model}")
            YOLO.model = load_model(latest_model)

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
    MODEL_FOLDER = "/app/"

    MODEL_CELLULES = (18,32) # Number of cells heigth, width (line,column)
    IMAGE_SIZE = (576, 1024) # Heigth, Width
    
    MODEL_ANCHOR_BOXES = [(7.771702728797546, 11.099425402414909), (16.233133106872444, 4.676295486695418), (26.65295286429116, 2.322343759539281), (9.476821452651933, 3.7753047321480744), (3.642066698646129, 5.355965495956248), (5.794567115957937, 2.5241804259133427), (3.5527940911190563, 2.2905150829398315), (1.7457785313564547, 2.229531520410109), (1.2712882556016014, 1.0891510755676512)]
    #MODEL_ANCHOR_BOXES = [(1.2593828217608534, 1.0782512065998489), (5.811058907896213, 2.5475515470119525), (25.958305961937022, 2.7822420043362435), (3.7260445560310216, 5.284102175173011), (3.5137867478357787, 2.28769758866406), (16.055079462929534, 4.655634908321043), (1.739537078536119, 2.2008474914889478), (7.646170972878436, 11.903683805213458), (9.540802666027334, 3.8905800088637648)]
    MODEL_CELLULES_SIZE = (IMAGE_SIZE[0]/MODEL_CELLULES[0], IMAGE_SIZE[1]/MODEL_CELLULES[1]) # Taille d'une cellule en pixels
    THRESHOLD_CONFIDENCE = 0.7 # Seuil de confiance pour la détection d'objet
    MODEL_MINIMAL_IOU = 0.5 # Seuil minimal d'IOU pour considérer une détection comme correcte
    MODEL_GRID_SENSIBILITY_COEF = 1.2 # Coefficient d'extension de la sigmoid pour x,y
    MODEL_SIGMOID_MULTIPLIER = 1 # Multiplicateur de la sigmoid pour w,h
    MODEL_SIGMOID_ADDER = 0.5 # Ajout à la sigmoid pour w,h
    
    LABEL_STUDIO_BASEURL = 'http://label-studio:8080'
    API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
    print(f'ML Backend | Clé API : {API_KEY}', flush=True)
    app = init_app(YOLO)
    app.run(host="0.0.0.0", port=9090, debug=True)