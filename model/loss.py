import torch.nn.functional as F
import numpy as np

def nll_loss(output, target):
    return F.nll_loss(output, target)

def nsrr_loss(output: np.ndarray, target: np.ndarray) -> float:
    # todo
    pass
