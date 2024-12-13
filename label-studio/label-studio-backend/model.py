from label_studio_ml.model import LabelStudioMLBase
import os
import glob
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model


UNLABELED_IMAGES_FOLDER = "/data/unlabeled/"  
OUTPUT_LABELS_FOLDER = "/data/labels/"
MODEL_FOLDER = "/models/"

class YOLO(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(YOLO, self).__init__(**kwargs)
        
        # Localiser et charger le fichier du modèle YOLO
        model_files = glob.glob(os.path.join(MODEL_FOLDER, "yolo_*.keras"))
        if not model_files:
            raise FileNotFoundError("Aucun modèle YOLO trouvé dans le dossier spécifié.")
        
        # Prendre le modèle avec la date la plus récente
        latest_model = max(model_files, key=os.path.getctime)
        print(f"Chargement du modèle : {latest_model}")
        self.model = load_model(latest_model)

    def _load_and_preprocess_image(self, image_path):
        input_shape = self.model.input.shape[1:3]  # (hauteur, largeur) attendues
        
        # Charger et prétraiter l'image
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # S'assurer que l'image est RGB
            img = img.resize((input_shape[1], input_shape[0]))  # Redimensionner
            image_array = np.array(img) / 255.0  # Normalisation entre 0 et 1
            return np.expand_dims(image_array, axis=0)  # Ajouter une dimension batch

    def predict(self, tasks, **kwargs):
        results = []
        for task in tasks:
            result_task = []
            image_path = task['data']['image']  # example "data:image1.jpg"
            image_path = image_path.replace("data:", UNLABELED_IMAGES_FOLDER)
            image = self._load_and_preprocess_image(image_path) # w/ batch
            pred = self.model.predict(image,verbose=1)
            # post process
            boxes = []

            # json
            for box in boxes :
                result_box = {}
                result_box['from_name'] = 'label'
                result_box['to_name'] = 'image'
                result_box['type'] = 'rectanglelabels'
                result_box['value'] = {
                                    'x': 40,
                                    'y': 35,
                                    'width': 20,
                                    'height': 30,
                                    'rectanglelabels': ['Object']
                    }
                result_task.append(result_box)
        
        results.append({
        'result': result_task
        })

        
        return results


