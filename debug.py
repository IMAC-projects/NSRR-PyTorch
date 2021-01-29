import argparse
import collections

import torch.autograd

from parse_config import ConfigParser
from data_loader import NSRRDataLoader

from utils.unit_test import UnitTest


def main(config):
    downscale_factor = config['data_loader']['args']['downsample']
    downscale_factor = [downscale_factor, downscale_factor]
    root_dir = config['data_loader']['args']['data_dir']
    batch_size = 8

    # UnitTest.dataloader_iteration(root_dir, batch_size)
    loader = NSRRDataLoader(root_dir=root_dir, batch_size=batch_size, downscale_factor=downscale_factor)
    # get a single batch
    x_view, x_depth, x_flow, _ = next(iter(loader))

    # Test util functions
    UnitTest.backward_warping(x_view, x_flow, downscale_factor)
    UnitTest.nsrr_loss(x_view)
    UnitTest.zero_upsampling(x_view, downscale_factor)

    # Test neural network
    UnitTest.feature_extraction(x_view, x_depth)
    rgbd = torch.cat((x_view, x_depth), 1)
    UnitTest.feature_reweight(rgbd, rgbd, rgbd, rgbd, rgbd)
    UnitTest.reconstruction(rgbd, rgbd)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='NSRR Unit testing')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='unused here')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-ds', '--downscale'], type=int, target=('data_loader', 'args', 'downsample'))
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
