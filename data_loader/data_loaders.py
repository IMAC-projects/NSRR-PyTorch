
import os
from base import BaseDataLoader

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as tf

from PIL import Image

from typing import Tuple


class NSRRDataLoader(BaseDataLoader):
    """

    """
    depth_dirname = "Depth"
    motion_dirname = "Motion"
    view_dirname  = "View"

    def __init__(self,
                 root_dir: str,
                 batch_size: int,
                 suffle: bool = True,
                 validation_split: float = 0.0,
                 num_workers: int = 1,
                 training: bool =True
                 ):
        dataset = NSRRDataset(root_dir,
                              NSRRDataLoader.view_dirname,
                              NSRRDataLoader.depth_dirname,
                              NSRRDataLoader.motion_dirname
                               )
        super(NSRRDataLoader, self).__init__(dataset,
                                             batch_size,
                                             suffle,
                                             validation_split,
                                             num_workers,
                                             training
                                             )


class NSRRDataset(Dataset):
    """
    Requires that corresponding view, depth and motion frames share the same name.
    """
    def __init__(self,
                 root_dir: str,
                 view_dirname: str,
                 depth_dirname: str,
                 motion_dirname: str,
                 transform: nn.Module = None,
                 downscale_factor: Tuple[int, int] = (2.0, 2.0)
                 ):
        super.__init__()
        self.root_dir = root_dir
        self.view_dirname = view_dirname
        self.depth_dirname = depth_dirname
        self.motion_dirname = motion_dirname

        self.downscale_factor = downscale_factor

        if transform is None:
            self.transform = tf.ToTensor()
        self.view_listdir = os.listdir(os.path.join(self.root_dir, self.view_dirname))

    def __getitem__(self, index):
        # view
        image_name = self.view_listdir[index]
        view_path = os.path.join(self.root_dir, self.view_dirname, image_name)
        depth_path = os.path.join(self.root_dir, self.depth_dirname, image_name)
        motion_path = os.path.join(self.root_dir, self.motion_dirname, image_name)

        trans = self.transform

        img_view_truth = trans(Image.open(view_path))

        # Infer size of the downscaled view, depth and optical flow.
        # bit dirty.
        downscaled_size = tuple((
            np.asarray(img_view_truth.size()[1:])
            / np.asarray(self.downscale_factor)
        ).astype(int).tolist())

        trans_downscale = tf.Resize(downscaled_size)
        trans = tf.Compose([trans_downscale, trans])

        img_view = trans_downscale(img_view_truth)
        # depth data is in a single-channel image.
        img_depth = trans(Image.open(depth_path).convert(mode="L"))
        img_flow = trans(Image.open(motion_path))

        return ((img_view, img_depth, img_flow), img_view_truth)

    def __len__(self) -> int:
        return len(self.view_listdir)

