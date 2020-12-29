from __future__ import division, print_function

import argparse
import logging
import os

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from classification import train_model
from dataset import Data_loader_3D
from resnet import generate_model

matplotlib.use('Agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(img_root_dir: str, dataloaders, image_datasets, num_class: int, num_epochs: int, model_depth: int, optimizer: str, lr: float, batch_size: int, weight_decay: float = 0):
    """
    Args:
        img_root_dir (str): the absolute path of the top directory of the whole dataset
        dataloaders (Dict[str: torch.utils.data.DataLoader]): dataloaders
        image_datasets (Dict[str: torch.utils.data.Dataset]): datasets
        num_class (int): the number of classes.
        num_epochs (int): the number of epochs to train the model.
        model (str): choose which kind of models to use.
        optimizer (str): choose which kind of optimizer to use.
        lr (float): learning rate
        batch_size (int): batch size
        weight_decay (float, optional): weight_decay(L2 penalty). Default is 0.
    Return:
        prefix (str): the prefix of the parameter file of the model.
    """
    assert model_depth in [10, 18, 34, 50, 101, 152,
                           200], "model_depth should be in [10, 18, 34, 50, 101, 152, 200], but got {}".format(model_depth)
    model = generate_model(model_depth)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    assert optimizer in [
        'sgd', 'adam'], 'Not supported optimizer type: {}, only support sgd and adam'.format(optimizer)
    if optimizer == 'sgd':
        optimizer_ft = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer_ft = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    prefix = 'resnet_3d_' + str(model_depth) + "_bs" + str(batch_size) + "_optim_" + optimizer + "_lr" + str(
        lr) + "_wd" + str(weight_decay) + "_epochs" + str(num_epochs) + "_data_" + img_root_dir.split('/')[-1]
    model = train_model(dataloaders, image_datasets, model,
                        criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs, prefix=prefix, num_class=num_class)
    torch.save(model, "./models/" + prefix + "_best_acc.pkl")
    return prefix


parser = argparse.ArgumentParser(
    description="Image classification using 3D ResNet")
parser.add_argument('--img_root_dir',
                    type=str, help='the absolute path of the top directory of the whole dataset')
parser.add_argument('--batch_size', default=4, type=int,
                    help='batch size, default 4')
parser.add_argument('--num_epochs', default=30, type=int,
                    help='the number of epochs to train the model, default 30')
parser.add_argument('--model_depth', default='10', type=int,
                    help='the depth of the ResNet, only support [10, 18, 34, 50, 101, 152, 200], default 10')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='optimizer to optimize the model parameters, default adam')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate, default 0.001')
parser.add_argument('--weight_decay', default=0,
                    type=float, help='weight_decay, default 0')

if __name__ == '__main__':
    args = parser.parse_args()
    num_class = 5
    img_root_dir = args.img_root_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    model_depth = args.model_depth
    optimizer = args.optimizer
    lr = args.lr
    weight_decay = args.weight_decay

    # check whether pwd has the needed subdirectories, if not make them
    for subdirs in ['images', 'logs', 'models']:
        if not os.path.exists(subdirs):
            os.makedirs(subdirs)

    model_name = 'resnet_3d_' + str(model_depth)

    log_name = './logs/' + model_name + '_bs' + str(batch_size) + '_optim_' + optimizer + '_lr' + str(
        lr) + '_wd' + str(weight_decay) + '_epochs' + str(num_epochs) + '_data_' + img_root_dir.split('/')[-1] + '.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_name,
                        filemode='w')
    logging.info('model: {} batch_size: {} optimizer: {} learning_rate: {:.4f} weight_decay: {:.4f} num_epochs: {} dataset: {}'.format(
        model_name, batch_size, optimizer, lr, weight_decay, num_epochs, img_root_dir))
    image_datasets, dataloaders = Data_loader_3D(
        img_root_dir=img_root_dir, batch_size=batch_size)
    prefix = train(img_root_dir, dataloaders, image_datasets, num_class,
                   num_epochs, model_depth, optimizer, lr, batch_size, weight_decay)
