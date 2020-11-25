import numpy as np
import matplotlib.pyplot as plt

class DetailPredict():
    
    def __init__(self, 
                 predict_label: int, 
                 predicted_image: np.array, 
                 original_label: int,
                 original_image: np.array,
                 confidence: float,
                 reconstruction_error: float,
                 components: int,
                 type_error: str = None):
        
        self._predict_label = predict_label
        self._predicted_image = predicted_image
        self._original_label = original_label
        self._original_image = original_image
        self._confidence = confidence
        self._reconstruction_error = reconstruction_error
        self._components = components
        self._type_error = type_error
    
    @property
    def reconstructed_image(self) -> np.array:
        return self._predicted_image.reshape(80, 80).T
    
    @property
    def original_image(self) -> np.array:
        return self._original_image.reshape(80, 80).T
    
    @property
    def type_error(self):
        return self._type_error
    
    @property
    def confidence(self):
        return self._confidence
    
    @property
    def reconstruction_error(self):
        return self._reconstruction_error
    
    @property
    def components(self):
        return self._components
    
    @property
    def okay_label(self):
        return self._predict_label == self._original_label
    
    def set_type_error(self, type_error):
        self._type_error = type_error
    
    def to_string(self):
        
        return f'Confidence: {round(self._confidence, 2)}\nReconstructed error: {round(self._reconstruction_error, 2)}\nComponents: {self._components}'
    
    def show_images(self):

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(6, 6)
        fig.suptitle(f'{self._type_error}', y=.82)

        ax1.imshow(self.reconstructed_image, cmap=plt.cm.bone)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f'Reconstructed {self._predict_label}')         
        ax1.text(0, -0.25, self.to_string(),
            verticalalignment='bottom', 
            horizontalalignment='left',
            transform=ax1.transAxes,
            color='black', fontsize=10)
        
        ax2.imshow(self.original_image, cmap=plt.cm.bone)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title(f'Original {self._original_label}')