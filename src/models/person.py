import numpy as np

class Person:
    
    def __init__(self, code: int, label: str, data: np.ndarray):
        self._code = code
        self._label = label
        self._data = data
      
    @property
    def code(self) -> int:
        return self._code
    
    @property
    def label(self) -> str:
        return self._label
    
    @property
    def data(self) -> np.array:
        return self._data