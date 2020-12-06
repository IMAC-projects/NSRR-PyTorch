import abc


class BaseDataset(abc.ABC):
    """
    Abstract base class for each data type.
    Child classes should implement __getitem__ and __len__,
    so that Pytorch may iterate over them.
    """

    @abc.abstractmethod
    def __getitem__(self, item):
        return NotImplemented

    @abc.abstractmethod
    def __len__(self):
        return NotImplemented

