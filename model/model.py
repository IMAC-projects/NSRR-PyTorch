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
            nn.Tanh()
        )
        self.add_module("weighting", process_seq)

    def mapRangeToRange(self, tensor, in_min = 0, in_max = 10, out_min = 0, out_max = 1):
        return torch.div(torch.mul(torch.add(tensor, -in_min), out_max - out_min), (in_max - in_min) + out_min)

    def forward(self, i0_rgbd_image: torch.Tensor, i1_rgbd_image: torch.Tensor,
                i2_rgbd_image: torch.Tensor, i3_rgbd_image: torch.Tensor,
                i4_rgbd_image: torch.Tensor) -> torch.Tensor:
        # Generates a pixel-wise weighting map for the current and each previous frames.
        # TODO cache the results
        i0_wmap = self.weighting(i0_rgbd_image)
        i0_wmap = self.mapRangeToRange(i0_wmap, -1, 1, 0, self.scale)
        i1_wmap = self.weighting(i1_rgbd_image)
        i1_wmap = self.mapRangeToRange(i1_wmap, -1, 1, 0, self.scale)
        i2_wmap = self.weighting(i2_rgbd_image)
        i2_wmap = self.mapRangeToRange(i2_wmap, -1, 1, 0, self.scale)
        i3_wmap = self.weighting(i3_rgbd_image)
        i3_wmap = self.mapRangeToRange(i3_wmap, -1, 1, 0, self.scale)

        # Each weighting map is multiplied to all features of the corresponding previous frame.
        i3_wmap = torch.mul(i3_wmap, i4_rgbd_image)
        i2_wmap = torch.mul(i2_wmap, i3_rgbd_image)
        i1_wmap = torch.mul(i1_wmap, i2_rgbd_image)
        i0_wmap = torch.mul(i0_wmap, i1_rgbd_image)

        # Weight multiply
        x = torch.mul(i0_wmap, i1_wmap)
        x = torch.mul(x, i2_wmap)
        x = torch.mul(x, i3_wmap)
        return x


class NSRRReconstructionModel(BaseModel):
    """
    """

    def __init__(self):
        super(NSRRReconstructionModel, self).__init__()
        padding = 1
        kernel_size = 3

        self.pool = nn.MaxPool2d(kernel_size, padding=padding, return_indices=True)
        self.upsize = nn.MaxUnpool2d(kernel_size, padding=padding)

        # Split the network into 5 groups of 2 layers to apply concat operation at each stage
        # Encoder 1 is symmetrical to Decoder 1   
        encoder1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        center = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )

        self.add_module("encoder_1", encoder1)
        self.add_module("encoder_2", encoder2)
        self.add_module("center", center)
        self.add_module("decoder_2", decoder2)
        self.add_module("decoder_1", decoder1)

    def forward(self, current_features: torch.Tensor, previous_features: torch.Tensor) -> torch.Tensor:
        # Features of the current frame and the reweighted features
        # of previous frames are concatenated
        x = torch.cat(current_features, previous_features)

        # 1. Cache result to handle 'skipped' connection for encoder 1
        x_encoder_1 = self.encoder_1(x)

        # 2. Apply pooling to reduce image size and store indices for future upsizing
        x, pool_1_indices = self.pool(x_encoder_1)

        # Process with step 1. and 2.
        x_encoder_2 = self.encoder_2(x)
        x, pool_2_indices = self.pool(x_encoder_2)

        # 3. Continue to run the model and upsize the image to progressively restore its original size
        x = self.center(x)
        x = self.upsize(x, pool_2_indices)
        
        # 4. We concatenate the original input that 'skipped' the network for encoder 1 and 2
        x = torch.cat(x, x_encoder_2)

        # Process with step 3. and 4.
        x = self.decoder_2(x)
        x = self.upsize(x, pool_1_indices)
        x = torch.cat(x, x_encoder_1)
        x = self.decoder_1(x)
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

        def _layer_forward_func(layer_index: int) -> Callable[[nn.Module, Any, Any], None]:
            def _layer_hook(module_: nn.Module, input_, output) -> None:
                self.output_layers[layer_index] = output
            return _layer_hook
        self.layer_forward_func = _layer_forward_func

        for name, module in self.model.named_children():
            if self.layer_predicate(name, module):
                module.register_forward_hook(
                    self.layer_forward_func(len(self.output_layers)))
                self.output_layers.append(torch.Tensor())

    def forward(self, x) -> List[torch.Tensor]:
        self.model(x)
        return self.output_layers
