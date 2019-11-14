from abc import ABC, abstractmethod
from src.output import Secret


class ObliviousSequence(ABC):
    @abstractmethod
    def len(self):
        pass

    @abstractmethod
    def map(self, function):
        pass

    @abstractmethod
    def sum(self):
        pass

    @abstractmethod
    def select(self, *include):
        pass
