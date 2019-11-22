from abc import ABC, abstractmethod


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
