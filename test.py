import argparse
import collections
import matplotlib.pyplot as plt

import torchvision
import torch.autograd
import torch.nn as nn
import torchvision.transforms as tf
import torch.nn.functional as F

from parse_config import ConfigParser
from model import LayerOutputModelDecorator, NSRRFeatureExtractionModel
from utils import Timer, upsample_zero_2d, optical_flow_to_motion, backward_warp_motion
from data_loader import NSRRDataLoader

from typing import Tuple


class UnitTest:

    @staticmethod
    def nsrr_loss(img_view: torch.Tensor) -> None:
        ## NSRR Loss
        vgg_model = torchvision.models.vgg16(pretrained=True, progress=True)
        vgg_model.eval()
        layer_predicate = lambda name, module: type(module) == nn.Conv2d
        lom = LayerOutputModelDecorator(vgg_model.features, layer_predicate)

        # Preprocessing image. for reference,
        # see: https://gist.github.com/jkarimi91/d393688c4d4cdb9251e3f939f138876e
        # with a few changes.
        dim = (224, 224)
        trans = tf.Compose([tf.Resize(dim)])
        img_loss = trans(img_view)
        img_loss = torch.autograd.Variable(img_loss)

        with Timer() as timer:
            output_layers = lom.forward(img_loss)
        print('(Perceptual loss) Execution time: ', timer.interval, ' s')

        print("(Perceptual loss) Output of Conv2 layers: ")
        for output in output_layers:
            print("size: ", output.size())


    @staticmethod
    def feature_extraction(img_view: torch.Tensor, img_depth: torch.Tensor) -> None:
        ## Feature extraction
        feature_model = NSRRFeatureExtractionModel()

        with Timer() as timer:
            feat = feature_model.forward(img_view, img_depth)
        print('(Feature extraction) Execution time: ', timer.interval, ' s')
        # some visualisation, not very useful since they do not represent a RGB-image, but well.

        trans = tf.ToPILImage()
        plt.imshow(trans(feat[0]))
        plt.draw()
        plt.pause(0.01)


    @staticmethod
    def zero_upsampling(img_view: torch.Tensor, scale_factor: Tuple[int, int]) -> None:
        ## Zero-upsampling
        with Timer() as timer:
            img_view_upsampled = upsample_zero_2d(img_view, scale_factor=scale_factor)
        print('(Zero-upsampling) Execution time: ', timer.interval, ' s')

        trans = tf.ToPILImage()
        plt.imshow(trans(img_view_upsampled[0]))
        plt.draw()
        plt.pause(0.01)


    @staticmethod
    def backward_warping(img_view: torch.Tensor, img_flow: torch.Tensor, scale_factor: Tuple[int, int]) -> None:
        ## First, zero-upsampling
        img_view_upsampled = upsample_zero_2d(img_view, scale_factor=scale_factor)
        # According to the article, bilinear interpolation of optical flow gives accurate enough results.
        img_flow_upsampled = F.interpolate(img_flow, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        # HSV-RGB conversion sensitivity depends on export!
        sensitivity = 0.1
        with Timer() as timer:
            img_motion = optical_flow_to_motion(img_flow_upsampled, sensitivity=sensitivity)
        print('(RGB to HSV conversion) Execution time: ', timer.interval, ' s')
        trans = tf.ToPILImage()
        with Timer() as timer:
            warped_view = backward_warp_motion(img_view_upsampled, img_motion)
        print('(Backward warping of view) Execution time: ', timer.interval, ' s')
        plt.imshow(trans(warped_view[0]))
        plt.draw()
        plt.pause(0.01)


    @staticmethod
    def dataloader_iteration(root_dir: str, batch_size: int) -> None:
        loader = NSRRDataLoader(root_dir=root_dir, batch_size=batch_size)

        for batch_idx, x in enumerate(loader):
            x_view, x_depth, x_flow = x[:3]
            y_truth = x[3]
            print(f"Batch #{batch_idx}, sizes:")
            print(f"  view:  {x_view.size()}")
            print(f"  depth: {x_depth.size()}")
            print(f"  flow:  {x_flow.size()}")
            print(f"  truth: {y_truth.size()}")



def main(config):
    downscale_factor = config['data_loader']['args']['downsample']
    downscale_factor = [downscale_factor, downscale_factor]
    root_dir = config['data_loader']['args']['data_dir']
    batch_size = 8

    # UnitTest.dataloader_iteration(root_dir, batch_size)
    loader = NSRRDataLoader(root_dir=root_dir, batch_size=batch_size, downscale_factor=downscale_factor)
    # get a single batch
    x_view, x_depth, x_flow, _ = next(iter(loader))
    UnitTest.backward_warping(x_view, x_flow, downscale_factor)
    # UnitTest.nsrr_loss(x_view)
    UnitTest.feature_extraction(x_view, x_depth)
    UnitTest.zero_upsampling(x_view, downscale_factor)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='NSRR Unit testing')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-ds', '--downscale'], type=int, target=('data_loader', 'args', 'downsample'))
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
