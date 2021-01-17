import torch.nn.functional as F
import numpy as np

import torchvision
import pytorch_ssim

def nll_loss(output, target):
    return F.nll_loss(output, target)

def nsrr_loss(output: np.ndarray, target: np.ndarray) -> float:
    # todo
    pass

class VGG16PartialModel:
    """
    An implementation of the image perceptual losses defined in:

    todo: syntax
    Justin Johnson, Alexandre Alahi, and Li Fei-Fei. 2016. Perceptual losses for real-time
style transfer and super-resolution. In European Conference on Computer Vision.
694–711.

    """

    ##
    # Init
    # model = torchvision.models.vgg16(pretrained=True, progress=True)
    # model.eval()


