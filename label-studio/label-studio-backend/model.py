from label_studio_ml.model import LabelStudioMLBase
UNLABELED_IMAGES_FOLDER = "/data/unlabeled"  
OUTPUT_LABELS_FOLDER = "/data/labels"

class YOLO(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(YOLO, self).__init__(**kwargs)
        # Chargez votre modèle ici (par exemple, un modèle TensorFlow ou PyTorch)

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            # Exemple de prédiction
            result = {
                'from_name': 'label',
                'to_name': 'image',
                'type': 'rectanglelabels',
                'value': {
                    'x': 10,
                    'y': 20,
                    'width': 30,
                    'height': 40,
                    'rectanglelabels': ['Object']
                }
            }
            predictions.append({'result': [result]})
        return predictions
