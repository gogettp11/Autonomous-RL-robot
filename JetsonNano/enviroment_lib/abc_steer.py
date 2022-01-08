from abc import ABC
from abc import abstractmethod

class Steer_abc(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def goRight(self, lenght : int):
        pass
    @abstractmethod
    def goLeft(self, lenght : int):
        pass
    @abstractmethod
    def goForward(self, lenght : int):
        pass