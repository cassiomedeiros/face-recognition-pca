#os
from typing import List

#Data structure
import numpy as np

#Viz
import matplotlib.pyplot as plt

#math
import math

#OpenCV
import cv2

#In-house library
from src.models.person import Person

class PCAEigenFaces():
    
    def __init__(self, number_components: int) -> None:
        
        self._number_components = number_components
        self._mean_image: np.array = []
        self._train: List[Person] = []
        self._diffs: np.array = []
        self._covariance: np.array =[]
        self._eigenvalues: np.array = []
        self._eigenvectors: np.array = []
        self._eigenfaces: np.array = []
        self._projections: np.array = []
        self._labels: List[int] = []
        
    @property
    def number_components(self) -> int:
        return self._number_components
     
    @property
    def mean_image(self) -> np.array:
        return self._mean_image
    
    @property 
    def train(self) -> List[Person]:
        return self._train
    
    @property
    def diffs(self) -> np.array:
        
        return self._diff
    
    @property
    def covariance(self) -> np.array:
        return self._covariance
    
    @property
    def eigenvectors(self) -> np.array:
        
        return self._eigenvectors
    
    @property
    def eigenvalues(self) -> np.array:
        
        return self._eigenvalues
    
    @property
    def eigenfaces(self) -> np.array:
        
        return self._eigenfaces
    
    @property
    def labels(self) -> List[int]:
        
        return self._labels
    
    @property
    def projections(self) -> np.array:
        
        return self._projections
    
    def _calc_mean_image(self) -> None:

        sample = self._train[0].data
        mean = np.zeros(sample.shape, dtype='float')
        data_size = len(self._train)

        for p in range(data_size):

            for i in range(len(mean)):

                mv = mean[i][0]
                pv = self.train[p].data[i][0]
                mv += pv

                mean[i][0] = mv

        self._mean_image =  mean / data_size
        
    def _calc_diff(self) -> None:
        
        sample = self._train[0].data
        data_size = len(self._train)   
        diffs = np.zeros((sample.shape[0], data_size), dtype='float')

        for i in range(diffs.shape[0]):
            for j in range(diffs.shape[1]):

                mean = self._mean_image[i][0]
                actual = self._train[j].data[i][0]
                diffs[i][j] = actual - mean

        self._diffs = diffs
        
    def _multiply(self, matrix_a: np.array, matrix_b) -> np.array:
    
        matrix_c = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))

        return cv2.gemm(matrix_a, matrix_b, 1, matrix_c, 1)
    
    def _cal_covariance(self) -> None:
    
        self._covariance =  self._multiply(self._diffs.T, self._diffs)
    
    def _calc_eigenvalues_eigenvectors(self) -> None:
        
        _, eigenvalues, eigenvectors = cv2.eigen(self._covariance)
        
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
    
    def _calc_eigenfaces(self) -> None:
        
        eigenvectors_t = self._eigenvectors.T
        eigenvectors_sel = eigenvectors_t[:,0:self._number_components]

        for j in range(self._number_components):
            eigenvectors_sel[:,j] = eigenvectors_t[:,j]

        eigenfaces = self._multiply(self._diffs, eigenvectors_sel)

        for j in range(eigenfaces.shape[1]):

            eigenfaces[:,j] = cv2.normalize(np.array(eigenfaces[:,j]), np.array(eigenfaces[:,j]))
            
        self._eigenfaces = eigenfaces
        
    def _cal_projections(self) -> None:
        
        labels: List[int] = [a for a in range(len(self._train))]
        projections: np.array = np.zeros((self._number_components, len(self._train)))

        for j in range(self._diffs.shape[1]):
            diff = self._diffs[:,j]
            w = self._multiply(self._eigenfaces.T, diff.reshape(-1, 1))
            projections[:,[j]] = w
            labels[j] = self._train[j].label
        
        self._labels = labels
        self._projections = projections
        
    def _calc_distance(self, matrix_a: np.array, matrix_b: np.array) -> float:
    
        total_distance: float = 0

        for i in range(matrix_a.shape[0]):
            distance = matrix_a[i][0] - matrix_b[i][0]
            total_distance += distance * distance

        return math.sqrt(total_distance)      

    def _calc_reconstruction(self, w: np.array) -> np.array:

        result = self._multiply(self._eigenfaces, w)

        result = cv2.add(result, self._mean_image)

        return result
        
    def fit(self, data: List[Person]) -> None:
        
        self._train = data
        self._calc_mean_image()
        self._calc_diff()
        self._cal_covariance()
        self._calc_eigenvalues_eigenvectors()
        self._calc_eigenfaces()
        self._cal_projections()
        
    def predict(self, test_data: np.array) -> (float, float, float, np.array):
    
        diff = cv2.subtract(test_data, np.uint64(self._mean_image))

        w = self._multiply(self._eigenfaces.T, np.float64(diff))
        min_j: int = 0
        min_distance = self._calc_distance(w, self._projections[:, [min_j]])

        for j in range(self._projections.shape[1]):

            distance = self._calc_distance(w, self._projections[:, [j]])

            if (distance < min_distance):
                min_distance = distance
                min_j = j

        label = self._labels[min_j]
        confidence = min_distance

        reconstruction = self._calc_reconstruction(w)
        reconstruction_error = cv2.norm(np.uint64(test_data), np.uint64(reconstruction), cv2.NORM_L2)
        
        return label, confidence, reconstruction_error, reconstruction
    
class PCAEigenFaces():
    
    def __init__(self, number_components: int):
        
        self._number_components = number_components
        self._mean_image: np.array = []
        self._train: List[Person] = []
        self._diffs: np.array = []
        self._covariance: np.array =[]
        self._eigenvalues: np.array = []
        self._eigenvectors: np.array = []
        self._eigenfaces: np.array = []
        self._projections: np.array = []
        self._labels: List[int] = []
        
    @property
    def number_components(self):
        return self._number_components
     
    @property
    def mean_image(self) -> np.array:
        return self._mean_image
    
    @property 
    def train(self) -> List[Person]:
        return self._train
    
    @property
    def diffs(self) -> np.array:
        
        return self._diff
    
    @property
    def covariance(self) -> np.array:
        return self._covariance
    
    @property
    def eigenvectors(self) -> np.array:
        
        return self._eigenvectors
    
    @property
    def eigenvalues(self) -> np.array:
        
        return self._eigenvalues
    
    @property
    def eigenfaces(self) -> np.array:
        
        return self._eigenfaces
    
    @property
    def labels(self) -> List[int]:
        
        return self._labels
    
    @property
    def projections(self) -> np.array:
        
        return self._projections
    
    def _calc_mean_image(self) -> None:

        sample = self._train[0].data
        mean = np.zeros(sample.shape, dtype='float')
        data_size = len(self._train)

        for p in range(data_size):

            for i in range(len(mean)):

                mv = mean[i][0]
                pv = self.train[p].data[i][0]
                mv += pv

                mean[i][0] = mv

        self._mean_image =  mean / data_size
        
    def _calc_diff(self) -> None:
        
        sample = self._train[0].data
        data_size = len(self._train)   
        diffs = np.zeros((sample.shape[0], data_size), dtype='float')

        for i in range(diffs.shape[0]):
            for j in range(diffs.shape[1]):
                mean = self._mean_image[i][0]
                actual = self._train[j].data[i][0]
                diffs[i][j] = actual - mean

        self._diffs = diffs
        
    def _multiply(self, matrix_a: np.array, matrix_b) -> np.array:
    
        matrix_c = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))

        return cv2.gemm(matrix_a, matrix_b, 1, matrix_c, 1)
    
    def _cal_covariance(self) -> None:
    
        self._covariance =  self._multiply(self._diffs.T, self._diffs)
    
    def _calc_eigenvalues_eigenvectors(self) -> None:
        
        _, eigenvalues, eigenvectors = cv2.eigen(self._covariance)
        
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
    
    def _calc_eigenfaces(self) -> None:
        
        eigenvectors_t = self._eigenvectors.T
        eigenvectors_sel = eigenvectors_t[:,0:self._number_components]

        for j in range(self._number_components):
            eigenvectors_sel[:,j] = eigenvectors_t[:,j]

        eigenfaces = self._multiply(self._diffs, eigenvectors_sel)

        for j in range(eigenfaces.shape[1]):

            eigenfaces[:,j] = cv2.normalize(np.array(eigenfaces[:,j]), np.array(eigenfaces[:,j]))
            
        self._eigenfaces = eigenfaces
        
    def _cal_projections(self) -> None:
        
        labels: List[int] = [a for a in range(len(self._train))]
        projections: np.array = np.zeros((self._number_components, len(self._train)))

        for j in range(self._diffs.shape[1]):
            diff = self._diffs[:,j]
            w = self._multiply(self._eigenfaces.T, diff.reshape(-1, 1))
            projections[:,[j]] = w
            labels[j] = self._train[j].label
        
        self._labels = labels
        self._projections = projections
        
    def _calc_distance(self, matrix_a: np.array, matrix_b: np.array) -> float:
    
        total_distance: float = 0

        for i in range(matrix_a.shape[0]):
            distance = matrix_a[i][0] - matrix_b[i][0]
            total_distance += distance * distance

        return math.sqrt(total_distance)      

    def _calc_reconstruction(self, w: np.array) -> np.array:

        result = self._multiply(self._eigenfaces, w)

        result = cv2.add(result, self._mean_image)

        return result
        
    def fit(self, data: List[Person]) -> None:
        
        self._train = data
        self._calc_mean_image()
        self._calc_diff()
        self._cal_covariance()
        self._calc_eigenvalues_eigenvectors()
        self._calc_eigenfaces()
        self._cal_projections()
        
    def predict(self, test_data: np.array) -> (float, float, float, np.array):
    
        diff = cv2.subtract(test_data, np.uint64(self._mean_image))

        w = self._multiply(self._eigenfaces.T, np.float64(diff))
        min_j: int = 0
        min_distance = self._calc_distance(w, self._projections[:, [min_j]])

        for j in range(self._projections.shape[1]):

            distance = self._calc_distance(w, self._projections[:, [j]])

            if (distance < min_distance):
                min_distance = distance
                min_j = j

        label = self._labels[min_j]
        confidence = min_distance

        reconstruction = self._calc_reconstruction(w)
        reconstruction_error = cv2.norm(np.uint64(test_data), np.uint64(reconstruction), cv2.NORM_L2)
        
        return label, confidence, reconstruction_error, reconstruction
