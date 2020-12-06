
import abc
import torch.nn


class BaseModel(torch.nn.Module):
    """
    Base class for all models.
    """

    def __init__(self, configuration):
        super(BaseModel, self).__init__()
        pass

    @abc.abstractmethod
    def forward(self, x):
        """
        Runs forward pass.
        """
        return NotImplemented

    @abc.abstractmethod
    def optimize_parameters(self):
        """
        Modifies weights after training pass.
        Basically computes gradients.
        """
        return NotImplemented

    @abc.abstractmethod
    def backward(self, y):
        pass

    @abc.abstractmethod
    def pre_epoch_callback(self):
        """
        Should be called before each epoch.
        Optional override.
        """
        pass

    @abc.abstractmethod
    def post_epoch_callback(self):
        """
        Should be called before each epoch.
        Optional override.
        """
        pass

    @abc.abstractmethod
    def test(self):
        """
        Should be called during validation.
        Optional override.
        """
        pass
