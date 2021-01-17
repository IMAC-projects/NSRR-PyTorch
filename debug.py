
import argparse
import torchvision
import torch.autograd
import torch.nn as nn
import torchvision.transforms as tf
from PIL import Image

from model import LayerOutputModelDecorator

def main(img: Image):
    vgg_model = torchvision.models.vgg16(pretrained=True, progress=True)
    vgg_model.eval()
    layer_predicate = lambda name, module: type(module) == nn.Conv2d
    lom = LayerOutputModelDecorator(vgg_model.features, layer_predicate)

    # Preprocessing image. for reference,
    # see: https://gist.github.com/jkarimi91/d393688c4d4cdb9251e3f939f138876e
    dim = (224, 224)
    trans = tf.Compose([tf.Resize(dim), tf.ToTensor()])
    img = trans(img)
    img.unsqueeze_(0)
    img = torch.autograd.Variable(img)

    output_layers = lom.forward(img)
    print("Output of Conv2 layers: ")
    for output in output_layers:
        print(output.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-i', '--input', required=True, type=str,
                      help='path to input image.')
    args = parser.parse_args()
    img = Image.open(args.input)

    main(img)
