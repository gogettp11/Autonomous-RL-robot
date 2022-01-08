from abc import ABC
from abc import abstractmethod

class Microphone_abc(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def returnFrequenciesMagnitudes(self):
        pass