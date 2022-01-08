from abc import ABC
from abc import abstractmethod

class Camera_abc(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def getImageRedPixelsCount(self):
        pass