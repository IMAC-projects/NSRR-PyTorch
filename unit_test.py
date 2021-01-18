
import os
import argparse

import matplotlib.pyplot as plt

from PIL import Image
import torchvision
import torch.autograd
import torch.nn as nn
import torchvision.transforms as tf


from model import LayerOutputModelDecorator, NSRRFeatureExtractionModel
from utils import upsample_zero_2d, Timer


def unit_test_loss(img_view: Image):
    ## NSRR Loss
    vgg_model = torchvision.models.vgg16(pretrained=True, progress=True)
    vgg_model.eval()
    layer_predicate = lambda name, module: type(module) == nn.Conv2d
    lom = LayerOutputModelDecorator(vgg_model.features, layer_predicate)

    # Preprocessing image. for reference,
    # see: https://gist.github.com/jkarimi91/d393688c4d4cdb9251e3f939f138876e
    # with a few changes.
    dim = (224, 224)
    trans = tf.Compose([tf.Resize(dim), tf.ToTensor()])
    img_loss = trans(img_view)
    img_loss.unsqueeze_(0)
    img_loss = torch.autograd.Variable(img_loss)

    with Timer() as timer:
        output_layers = lom.forward(img_loss)
    print('(Perceptual loss) Execution time: ', timer.interval, ' s')

    print("(Perceptual loss) Output of Conv2 layers: ")
    for output in output_layers:
        print(output.shape)


def unit_test_feature_extraction(img_view: Image, img_depth: Image):
    ## Feature extraction
    trans = tf.Compose([tf.ToTensor()])
    img_view = trans(img_view)
    img_depth = trans(img_depth)
    img_view.unsqueeze_(0)
    img_depth.unsqueeze_(0)
    feature_model = NSRRFeatureExtractionModel()

    feat = feature_model.forward(img_view, img_depth)
    # some visualisation, not very useful since they do not represent a RGB-image, but well.
    trans = tf.ToPILImage()
    plt.imshow(trans(feat[0]))
    plt.draw()
    plt.pause(0.01)


def unit_test_zero_upsampling(img_view: Image):
    ## Zero-upsampling
    trans = tf.Compose([tf.ToTensor()])
    img_view = trans(img_view)
    img_view.unsqueeze_(0)

    scale_factor = (2.0, 2.0)

    with Timer() as timer:
        img_view_upsampled = upsample_zero_2d(img_view, scale_factor=scale_factor)
    print('(Zero-upsampling) Execution time: ', timer.interval, ' s')

    print(img_view_upsampled.size())
    trans = tf.ToPILImage()
    plt.imshow(trans(img_view_upsampled[0]))
    plt.draw()
    plt.pause(0.01)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug')
    parser.add_argument('-d', '--directory', required=True, type=str,
                      help='path to input directory.')
    parser.add_argument('-n', '--filename', required=True, type=str,
                        help='shared name of the view, depth, and motion files.')

    args = parser.parse_args()
    img_view = Image.open(os.path.join(args.directory, "View", args.filename))
    img_depth = Image.open(os.path.join(args.directory, "Depth", args.filename))
    # depth data is in a single-channel image.
    img_depth = img_depth.convert(mode="L")
    img_motion = Image.open(os.path.join(args.directory, "Motion", args.filename))

    # unit_test_loss(img_view)
    # unit_test_feature_extraction(img_view, img_depth)
    unit_test_zero_upsampling(img_view)
