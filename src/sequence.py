from abc import ABC, abstractmethod


class ObliviousSequence(ABC):
    @abstractmethod
    def len(self):
        pass

    @abstractmethod
    def map(self, function):
        pass

    @abstractmethod
    def reduce(self, neutral_element, operation, initial=None):
        pass

    @abstractmethod
    def sum(self):
        pass

    @abstractmethod
    def select(self, *include):
        pass

    @abstractmethod
    def is_included(self, index):
        pass
