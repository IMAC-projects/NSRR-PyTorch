
import os
import argparse

import matplotlib.pyplot as plt

from PIL import Image
import torchvision
import torch.autograd
import torch.nn as nn
import torchvision.transforms as tf
import torch.nn.functional as F


from model import LayerOutputModelDecorator, NSRRFeatureExtractionModel
from utils import Timer, upsample_zero_2d, optical_flow_to_motion, backward_warp_motion
from data_loader import NSRRDataLoader


def unit_test_loss(img_view: torch.Tensor):
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
        print(output.shape)


def unit_test_feature_extraction(img_view: torch.Tensor, img_depth: torch.Tensor):
    ## Feature extraction
    feature_model = NSRRFeatureExtractionModel()

    feat = feature_model.forward(img_view, img_depth)
    # some visualisation, not very useful since they do not represent a RGB-image, but well.
    trans = tf.ToPILImage()
    plt.imshow(trans(feat[0]))
    plt.draw()
    plt.pause(0.01)


def unit_test_zero_upsampling(img_view: torch.Tensor):
    ## Zero-upsampling
    scale_factor = (2.0, 2.0)

    with Timer() as timer:
        img_view_upsampled = upsample_zero_2d(img_view, scale_factor=scale_factor)
    print('(Zero-upsampling) Execution time: ', timer.interval, ' s')

    print(img_view_upsampled.size())
    trans = tf.ToPILImage()
    plt.imshow(trans(img_view_upsampled[0]))
    plt.draw()
    plt.pause(0.01)


def unit_test_backward_warping(img_view: torch.Tensor, img_flow: torch.Tensor):
    ## First, zero-upsampling
    scale_factor = (2.0, 2.0)
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug')
    parser.add_argument('-d', '--directory', required=True, type=str,
                      help='path to input directory.')
    parser.add_argument('-n', '--filename', required=True, type=str,
                        help='shared name of the view, depth, and motion files.')

    args = parser.parse_args()

    # todo: use the DataLoader
    """
    batch_size = 2
    loader = NSRRDataLoader(args.directory, batch_size=batch_size)
    for y_truth, x_view, x_depth, x_flow in enumerate(loader):
        print(y_truth.size())
    """

    img_view = Image.open(os.path.join(args.directory, "View", args.filename))
    img_depth = Image.open(os.path.join(args.directory, "Depth", args.filename))
    # depth data is in a single-channel image.
    img_depth = img_depth.convert(mode="L")
    img_flow = Image.open(os.path.join(args.directory, "Motion", args.filename))

    trans = tf.Compose([tf.ToTensor()])

    img_view = trans(img_view)
    img_depth = trans(img_depth)
    img_flow = trans(img_flow)
    img_view.unsqueeze_(0)
    img_depth.unsqueeze_(0)
    img_flow.unsqueeze_(0)

    # unit_test_loss(img_view)
    # unit_test_feature_extraction(img_view, img_depth)
    # unit_test_zero_upsampling(img_view)
    unit_test_backward_warping(img_view, img_flow)
