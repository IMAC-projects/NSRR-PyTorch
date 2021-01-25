import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from typing import List, Callable, Any


class NSRRFeatureExtractionModel(BaseModel):
    """
    """
    def __init__(self):
        super(NSRRFeatureExtractionModel, self).__init__()
        kernel_size = 3
        # Adding padding here so that we do not lose width or height because of the convolutions.
        # The input and output must have the same image dimensions so that we may concatenate them
        padding = 1
        process_seq = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.add_module("featuring", process_seq)

    def forward(self, colour_images: torch.Tensor, depth_images: torch.Tensor) -> torch.Tensor:
        # From each 3-channel image and 1-channel image, we construct a 4-channel input for our model.
        x = torch.cat((colour_images, depth_images), 1)
        x_features = self.featuring(x)
        # We concatenate the original input that 'skipped' the network.
        x = torch.cat((x, x_features), 1)
        return x


class NSRRFeatureReweightingModel(BaseModel):
    """
    """
    def __init__(self):
        super(NSRRFeatureReweightingModel, self).__init__()
        # According to the paper, rescaling in [0, 10] after the final tanh activation
        # gives accurate enough results.
        self.scale = 10
        kernel_size = 3
        # Adding padding here so that we do not lose width or height because of the convolutions.
        # The input and output must have the same image dimensions so that we may concatenate them
        padding = 1
        process_seq = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 4, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            # todo: normalisation to [0, self.scale]
        )
        self.add_module("weighting", process_seq)

    def forward(self, colour_images: torch.Tensor, depth_images: torch.Tensor) -> torch.Tensor:
        # todo
        pass

class NSRRReconstructionModel(BaseModel):
    """
    """
    def __init__(self):
        # TODO check the padding and kernel_size
        padding = 1
        kernel_size = 3
        # TODO how to handle the "skip connection" with a concat
        # TODO what about the pooling ? and Upsize ?
        process_seq = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
            nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.add_module("reconstruction", process_seq)

    def forward(self, colour_images: torch.Tensor, depth_images: torch.Tensor) -> torch.Tensor:
        # todo
        pass


class LayerOutputModelDecorator(BaseModel):
    """
    A Decorator for a Model to output the output from an arbitrary set of layers.
    """

    def __init__(self, model: nn.Module, layer_predicate: Callable[[str, nn.Module], bool]):
        super(LayerOutputModelDecorator, self).__init__()
        self.model = model
        self.layer_predicate = layer_predicate

        self.output_layers = []

        def _layer_forward_func(layer_index: int) -> Callable[[nn.Module, Any, Any], None]:
            def _layer_hook(module_: nn.Module, input_, output) -> None:
                self.output_layers[layer_index] = output
            return _layer_hook
        self.layer_forward_func = _layer_forward_func

        for name, module in self.model.named_children():
            if self.layer_predicate(name, module):
                module.register_forward_hook(self.layer_forward_func(len(self.output_layers)))
                self.output_layers.append(torch.Tensor())

    def forward(self, x) -> List[torch.Tensor]:
        self.model(x)
        return self.output_layers


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
