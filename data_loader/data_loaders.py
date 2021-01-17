import os
import torch
import numpy as np
from imageio import imread
from torch.utils.data import Dataset
from base import BaseDataLoader


class NSRRDataLoader(BaseDataLoader):
    """

    """
    depth_dirname  = "Depth"
    motion_dirname = "Motion"
    view_dirname   = "View"

    def __init__(self,
                 root_dir,
                 batch_size,
                 suffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 training=True
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

    def _load_files(self):
        pass

class NSRRDataset(Dataset):
    """
    Requires that corresponding view, depth and motion frames share the same name.
    """
    def __init__(self,
                 root_dir,
                 view_dirname,
                 depth_dirname,
                 motion_dirname,
                 downscale_transform=None,
                 ):
        self.root_dir = root_dir
        self.view_dirname = view_dirname
        self.depth_dirname = depth_dirname
        self.motion_dirname = motion_dirname

        if not downscale_transform:
            self.downcale_transform = NSRRDataset._downscale

        self.view_listdir = os.listdir(os.path.join(self.root_dir, self.view_dirname))

    def __len__(self) -> int:
        return len(self.view_listdir)

    def __getitem__(self, index):
        # view
        image_name = self.view_listdir[index]
        view_path = os.path.join(self.root_dir, self.view_dirname, image_name)
        depth_path = os.path.join(self.root_dir, self.depth_dirname, image_name)
        motion_path = os.path.join(self.root_dir, self.motion_dirname, image_name)

        view_image = imread(view_path)
        depth_image = imread(depth_path)
        motion_image = imread(motion_path)

        downscaled_view_image = self.downcale_transform(view_image)

        return downscaled_view_image, depth_image, motion_image, view_image

    @staticmethod
    def _downscale(image) -> np.ndarray:
        pass




