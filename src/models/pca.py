from typing import List
#data structure
import pandas as pd
import numpy as np

#OpenCV
import cv2

#OS
import glob
import random
import math
from sys import float_info
import os

#In-house library
from src.models.person import Person
from src.models.detail_predict import DetailPredict
from src.models.eigen_faces import PCAEigenFaces

class PCA():
    
    def __init__(self, 
                 images_path: str, 
                 min_components: int = 15, 
                 max_components: int = 15, 
                 test_size: float = .3):
        
        self._images_path = images_path
        self._min_components = min_components
        self._max_components = max_components
        self._test_size = test_size
        self._dataset: pd.DataFrame = pd.DataFrame([])
        
    @property
    def dataset(self) -> List[Person]:
        
        return self._dataset
    
    @property
    def details_prediction(self) -> List[DetailPredict]:
        
        return self._details_prediction
    
    @property
    def result_by_component(self) -> pd.DataFrame:
        
        data: np.array = self._result_by_component
        
        return pd.DataFrame(data, columns=['Components', 'Accuracy'])
    
    def _get_image(self, file):
    
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE) 
        resized = cv2.resize(img, (80, 80), interpolation = cv2.INTER_AREA)
        resized_t = resized.T.reshape(resized.shape[1] * resized.shape[0], 1)

        img_result = np.uint64(resized_t)
    
        return img_result 
        
    def _load_images(self) -> None:
        
        files = [f for f in glob.glob(f"{self._images_path}*.jpg")]
        persons: List[Person] = []

        for file in files:
            basename = os.path.basename(file)[:-4].split('_')

            img = self._get_image(file)
            code = int(basename[0])
            label = int(basename[1])

            persons.append(Person(code=code, label=label, data=img))
        
        self._dataset = persons
        
    
    def _train_test_split(self) -> (List[Person], List[Person]):
        
        total_size = len(self._dataset)
        test_size = round(total_size * self._test_size)
        test_list_random = random.sample(range(1, total_size), test_size)

        test: List[Person] = [person for person in self._dataset if person.code in test_list_random]
        train: List[Person] = [person for person in self._dataset if person.code not in test_list_random]
            
        return train, test
    
    def processing(self):
        
        #to compute metrics
        min_distance: float = float_info.min
        max_distance: float = float_info.max
        mean_distance: int = 0
        corrects: int = 0
            
        min_rec: float = float_info.min
        max_rec: float = float_info.max
        mean_rec: float = 0
            
        MAX_DISTANCE: float = 2500
        MAX_REC: float = 2900
        #
        
        self._load_images()
        train, test = self._train_test_split()
        
        result_by_component: np.array = []
        details_prediction: List[DetailPredict] = []
        
        for components in np.arange(self._min_components, self._max_components+1):
            
            model = PCAEigenFaces(components)
            model.fit(train)
        
            true_positive_count: int = 0
            true_negative_count: int = 0
                
            for person_test in test:

                label, confidence, reconstruction_error, image_reconstructed = model.predict(person_test.data)
                
                okay_label = label == person_test.label
                
                if (okay_label):
                    corrects += 1
                    
                detail = DetailPredict(type_error=None,
                            predict_label=label, 
                            predicted_image=image_reconstructed,
                            original_label=person_test.label,
                            original_image=person_test.data,
                            confidence=confidence,
                            reconstruction_error=reconstruction_error,
                            components=components)
                    
                if reconstruction_error > MAX_REC:
                    
                    print(f"""NOT A PERSON - Predicted label: {label}, confidence: {confidence}, reconstructed error: {reconstruction_error}, original label: {person_test.label}""")
                    detail.set_type_error('NOT A PERSON')
                    if (not okay_label):
                        true_negative_count += 1
                        
                elif (confidence > MAX_DISTANCE):
                   
                    print(f"""UNKNOW PERSON (by distance) - Predicted label: {label}, confidence: {confidence}, reconstructed error: {reconstruction_error}, original label: {person_test.label}""")
                    detail.set_type_error('UNKNOW PERSON (by distance)')
                    if (not okay_label):
                        true_negative_count += 1
                        
                elif(reconstruction_error > 2400 and confidence > 1800):
                    
                    print(f"""UNKNOW PERSON (by two factors) - Predicted label: {label}, confidence: {confidence}, reconstructed error: {reconstruction_error}, original label: {person_test.label}""")
                    detail.set_type_error('UNKNOW PERSON (by two factors)')
                    if (not okay_label):
                        true_negative_count += 1
                        
                elif(okay_label):
                    true_positive_count += 1
                    detail.set_type_error('KNOWN')
                else:
                    print(f"""UNKNOW - Predicted label: {label}, confidence: {confidence}, reconstructed error: {reconstruction_error}, original label: {person_test.label}""")
                    detail.set_type_error('UNKNOW')
                    
                details_prediction.append(detail)
                    
                if (person_test.label <= 41):
                    
                    if confidence < min_distance:
                        min_distance = confidence
                        
                    if confidence > max_distance:
                        max_distance = confidence
                        
                    mean_distance += confidence
                    
                    if reconstruction_error < min_rec:
                        min_rec = reconstruction_error
                        
                    if (reconstruction_error > max_rec):
                        max_rec = reconstruction_error
                        
                    mean_rec += reconstruction_error
        
            
            total_size_test = len(test)
            trues = true_negative_count + true_positive_count
            accuracy = round(trues / total_size_test * 100, 2)
            
            result_by_component.append([components, accuracy])
            
            print(f"Number components: {components}, accuracy: {accuracy}% ({true_positive_count} of {total_size_test})")
            print(f"True positive count: {true_positive_count}, True negative count {true_negative_count}")
            print(f"Min distance: {min_distance}, Max distance: {max_distance}, Mean distance: {mean_distance / total_size_test}")
            print(f"Min rec: {min_rec}, Max rec: {max_rec}, Mean rec: {mean_rec / total_size_test}")
            print(f"Corrects: {corrects}")
        self._result_by_component = result_by_component
        self._details_prediction = details_prediction