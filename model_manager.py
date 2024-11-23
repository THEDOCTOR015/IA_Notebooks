import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


class ModelManager:
    def __init__(self, model_type, dataset_name, base_dir='models'):
        assert model_type is not None, 'Model type is not defined'
        assert dataset_name is not None, 'Dataset name is not defined'

        # Model name and path
        self.model_name = None
        self.model_path = None
        self.base_dir = base_dir
        
        # Metadata
        self.metadata = { 'model_type': model_type }
        self.metadata['training_history'] = []
        self.metadata['validation_history'] = []
        self.metadata['dataset_name'] = dataset_name
        self.metadata['date'] = datetime.now().strftime('%d-%m-%y')  # DD-MM-YY format
        self.metadata['date_time'] = datetime.now().strftime('%d-%m-%y %H:%M:%S')  # DD-MM-YY_HH-MM-SS format
        
        # Losses
        self.training_loss = []
        self.validation_loss = []
    
    def update_metadata(self):
        """
        Update the metadata for the model.
        """
        with open(os.path.join(self.model_path, f'{self.model_name}_metadata.json'), 'w') as json_file:
            json.dump(self.metadata, json_file, indent=4)

    def human_readable_count(self, num):
        """
        Converts a number into a human-readable format (e.g., 1.2K, 350M, 1.5B)
        """
        for unit in ['','k','M','B']:
            if abs(num) < 1000:
                return f"{int(num)}{unit}"
            num /= 1000.0
        raise ValueError('Parameter count is too large')

    def save_model(self, model, description=None):
        """
        Save a model along with its metadata in a versioned directory format.
        """
        param_count = self.human_readable_count(model.count_params())
        if self.model_name is None or self.model_path is None:
            model_name = f"{self.metadata['model_type']}_{self.metadata['dataset_name']}_{param_count}_{self.metadata['date']}"
            folder_path = os.path.join(self.base_dir, os.path.join(self.metadata['model_type'], model_name))
            os.makedirs(folder_path, exist_ok=True)
            self.model_name = model_name
            self.model_path = folder_path
        # Save model
        model_file_path = os.path.join(self.model_path, self.model_name)
        model.save(model_file_path+'.keras')
        
        # Save metadata
        self.metadata['model_name'] = self.model_name
        self.metadata['model_file'] = model_file_path+'.keras'
        self.metadata['param_count'] = param_count
        self.metadata['description'] = description if description is not None else self.metadata['description']
        self.update_metadata()

    def load_model(self, model_path, custom_objects=None):
        """
        Load a model with optional custom objects.
        """
        self.metadata = self.get_metadata(model_path)
        return load_model(model_path, custom_objects=custom_objects)

    def get_metadata(self, model_path):
        """
        Retrieve the metadata for a model.
        """
        metadata_json = [element for element in os.listdir(model_path) if element.endswith('metadata.json')][0]
        metadata_path = os.path.join(model_path, metadata_json)
        with open(metadata_path, 'r') as json_file:
            metadata = json.load(json_file)
        return metadata
    
    def add_loss(self, loss_value, loss_type='train'):
        """
        Add a loss value to the training or validation history.
        """
        if loss_type == 'train':
            self.training_loss.append(loss_value)
        elif loss_type == 'val':
            self.validation_loss.append(loss_value)
        else:
            raise ValueError('Invalid loss type. Use "train" or "val".')

    def update_losses(self):
        """
        Update the training and validation losses in the metadata.
        """
        if self.training_loss is not None:
            self.metadata['training_history'].append(self.training_loss)
        if self.validation_loss is not None:
            self.metadata['validation_history'].append(self.validation_loss)
        self.update_metadata()
    
    def smooth_curve(points, factor):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    def plot_losses(self, smoothing_factor=0.5, for_file=False):
        if 'training_history' in self.metadata:
            training_loss = [loss_element for loss_epoch in self.metadata['training_history'] for loss_element in loss_epoch]
            plt.plot(self.smooth_curve(training_loss, smoothing_factor), label='Smoothed Training Loss')
        if 'validation_history' in self.metadata:
            validation_loss = [loss_element for loss_epoch in self.metadata['validation_history'] for loss_element in loss_epoch]
            plt.plot(self.smooth_curve(validation_loss, smoothing_factor), label='Smoothed Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        if for_file:
            plt.savefig('training_validation_loss_plot.png')
        plt.show()