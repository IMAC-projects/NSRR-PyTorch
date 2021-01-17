
import os
import argparse
import torchvision
import torch.autograd
import torch.nn as nn
import torchvision.transforms as tf
from PIL import Image

from model import LayerOutputModelDecorator, NSRRFeatureExtractionModel


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

    output_layers = lom.forward(img_loss)
    print("Output of Conv2 layers: ")
    for output in output_layers:
        print(output.shape)

def unit_test_feature_extraction(img_view: Image, img_depth: Image):
    ## Feature extraction
    trans = tf.Compose([tf.ToTensor()])
    img_view = trans(img_view)
    img_depth = trans(img_depth)
    feature_model = NSRRFeatureExtractionModel()

    feat = feature_model.forward(img_view, img_depth)
    print(feat)


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

    unit_test_loss(img_view)
    # unit_test_feature_extraction(img_view, img_depth)
