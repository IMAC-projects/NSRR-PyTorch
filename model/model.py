
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from typing import List, Callable


class NSRRFeatureExtractionModel(BaseModel):
    """
    """
    def __init__(self):
        super(NSRRFeatureExtractionModel, self).__init__()
        process_seq = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3),
            nn.ReLU()
        )
        self.add_module("process_sequential", process_seq)

    def forward(self, img_colour: torch.Tensor, img_depth: torch.Tensor) -> torch.Tensor:
        # From a 3-channel image and a 1-channel image, we construct a 4-channel input for our model.
        print(img_colour.shape)
        print(img_depth.shape)
        x = torch.cat((img_colour, img_depth), 1)
        x_processed = self.process_sequetial(x)
        # We concatenate the original input that 'skipped' the network.
        x = torch.cat((x, x_processed), 0)
        return x


class LayerOutputModelDecorator(BaseModel):
    """
    A Decorator for a Model to output the output from an arbitrary set of layers.
    """
    def __init__(self, model: nn.Module, layer_predicate: Callable[[str, nn.Module], bool]):
        super(LayerOutputModelDecorator, self).__init__()
        self.model = model
        self.layer_predicate = layer_predicate

        self.output_layers = []

        def _layer_hook(module_: nn.Module, input_, output: torch.Tensor, layer_index) -> None:
            self.output_layers[layer_index] = output

        self.layer_forward_func = \
            lambda layer_index: \
                lambda module, input, output: \
                    _layer_hook(module, input, output, layer_index)

        for name, module in self.model.named_children():
            if self.layer_predicate(name, module):
                module.register_forward_hook(self.layer_forward_func(len(self.output_layers)))
                self.output_layers.append(None)

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

