import torch

import pytorch_ssim
from utils import PerceptualLossManager


def feature_reconstruction_loss(conv_layer_output: torch.Tensor, conv_layer_target: torch.Tensor) -> torch.Tensor:
    """
    Computes Feature Reconstruction Loss as defined in Johnson et al. (2016)
    todo: syntax
    Justin Johnson, Alexandre Alahi, and Li Fei-Fei. 2016. Perceptual losses for real-time
    style transfer and super-resolution. In European Conference on Computer Vision.
    694–711.
    Takes the already-computed output from the VGG16 convolution layers.
    """
    if conv_layer_output.shape != conv_layer_target:
        raise ValueError("Output and target tensors have different dimensions!")
    loss = conv_layer_output.dist(conv_layer_target, p=2) / torch.numel(conv_layer_output)
    return loss


def nsrr_loss(output: torch.Tensor, target: torch.Tensor, w: float) -> torch.Tensor:
    """
    Computes the loss as defined in the NSRR paper.
    """
    loss_ssim = 1 - pytorch_ssim.ssim(output, target)
    loss_perception = 0
    conv_layers_output = PerceptualLossManager.get_vgg16_conv_layers_output(output)
    conv_layers_target = PerceptualLossManager.get_vgg16_conv_layers_output(output)
    for i in range(len(conv_layers_output)):
        loss_perception += feature_reconstruction_loss(conv_layers_output[i], conv_layers_target[i])
    loss = loss_ssim + w * loss_perception
    return loss

