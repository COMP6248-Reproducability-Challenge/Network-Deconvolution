import argparse
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch Training')
    parser.add_argument('--epochs', '-e', default=100, type=int, help='training epochs')
    parser.add_argument('--batch-size', '-b', default=128, type=int, help='batch size')
    parser.add_argument('--dataset', '-d', default='cifar10', type=str, help='cifar10|cifar100')
    parser.add_argument('--arch', '-a', default='vgg16', type=str, help='vgg16|resnet18')
    parser.add_argument('--optimizer', '-o', default='SGD', type=str, help='SGD|Adam')

    parser.add_argument('--deconv', default=True, type=strtobool, help='use deconv')
    parser.add_argument('--delinear', default=True, type=strtobool, help='use deliner')

    parser.add_argument('--log', default=True, type=strtobool, help='log the result')
    parser.add_argument('--tensorboard', default=True, type=strtobool, help='use tensorboard')
    parser.add_argument('--save-model', default=True, type=strtobool, help='save the model')
    parser.add_argument('--save-dir', default='checkpoints', type=str, help='root dir to save checkpoints')
    return parser.parse_args()