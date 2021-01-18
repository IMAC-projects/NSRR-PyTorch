import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import time
import torch
import torchvision.transforms.functional as F
import numpy as np

from threading import Lock, Thread


def upsample_zero_2d(input: torch.Tensor, size=None, scale_factor=None) -> torch.Tensor:
    """
    IMPORTANT: we only support integer scaling factors for now!!
    """
    # input shape is: batch x channels x height x width
    # output shape is:
    if size is not None and scale_factor is not None:
        raise ValueError("Should either define both size and scale_factor!")
    if size is None and scale_factor is None:
        raise ValueError("Should either define size or scale_factor!")
    input_size = torch.tensor(input.size(), dtype=torch.int)
    input_image_size = input_size[2:]
    data_size = input_size[:2]
    if size is None:
        # Get the last two dimensions -> height x width
        # compare to given scale factor
        b_ = np.asarray(scale_factor)
        b = torch.tensor(b_)
        # check that the dimensions of the tuples match.
        if len(input_image_size) != len(b):
            raise ValueError("scale_factor should match input size!")
        output_image_size = (input_image_size * b).type(torch.int)
    else:
        output_image_size = size
    if scale_factor is None:
        scale_factor = output_image_size / input_image_size
    else:
        scale_factor = torch.tensor(np.asarray(scale_factor), dtype=torch.int)
    ##
    output_size = torch.cat((data_size, output_image_size))
    output = torch.zeros(tuple(output_size.tolist()))
    ##
    output[:, :, ::scale_factor[0], ::scale_factor[1]] = input
    return output


class SingletonPattern(type):
    """
    see: https://refactoring.guru/fr/design-patterns/singleton/python/example
    """
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

class Timer:
    """
    see: https://saladtomatonion.com/blog/2014/12/16/mesurer-le-temps-dexecution-de-code-en-python/
    """
    def __init__(self):
        self.start_time = None
        self.interval = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.interval = None
        self.start_time = time.time()

    def stop(self):
        if self.start_time is not None:
            self.interval = time.time() - self.start_time
            self.start_time = None

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
