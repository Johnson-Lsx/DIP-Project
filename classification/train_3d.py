from __future__ import division, print_function

import argparse
import logging
import os
import time
from tqdm import tqdm
import copy
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from classification import plot_curve
from dataset import Data_loader_3D
from metric import (cal_metric, sklearn_cal_metric, sklearn_plot, sklearn_stat,
                    stat)
from resnet import generate_model

matplotlib.use('Agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(dataloaders, image_datasets, model, criterion, optimizer, scheduler, num_epochs, prefix, num_class):
    """
    Args:
        dataloaders (Dict[str: torch.utils.data.DataLoader]): dataloaders that send the data to the model.
        image_datasets (Dict[str: torch.utils.data.Dataset]): training and validation datasets.
        model (torchvision.models): the network model to train.
        criterion (torch.nn.Loss Functions): loss functions.
        optimizer (torch.optim.Optimizer): optimizer to optimize the model parameters.
        scheduler (torch.optim.lr_scheduler): learning rate scheduler.
        num_epochs (int): the number of epochs to train the model.
        prefix (str): the name of images
        num_class (int): the number of classes.
    Returns:
        torchvision.models: the trained network model.
    """
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    cnt = 0
    use_auc = True
    train_acc = []
    dev_acc = []
    train_ls = []
    dev_ls = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            old_stat = dict()
            stat_list = list()
            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                inputs = torch.unsqueeze(inputs, 1)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

                old_stat = sklearn_stat(
                    outputs=outputs, labels=labels, old_stat=old_stat, use_auc=use_auc)
                stat_list = stat(num_classes=num_class, preds=preds,
                                 labels=labels, old_stat=stat_list)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            metric = sklearn_cal_metric(
                old_stat, use_auc=use_auc)
            metric_dict = cal_metric(stat=stat_list)
            if phase == 'train':
                train_acc.append(epoch_acc)
                train_ls.append(epoch_loss)
                scheduler.step()
            else:
                dev_acc.append(epoch_acc)
                dev_ls.append(epoch_loss)
            if use_auc:
                msg = 'epoch: {} phase: {} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1-score: {:.4f} AUC: {:.4f}'.format(
                    epoch, phase, epoch_loss, epoch_acc, metric['precision'], metric['recall'], metric['F1_score'], metric['auc'])
                print(msg)
                logging.info(msg)
                msg1 = 'The accuracy for class 0 is: {:.4f}, class 1 is: {:.4f}, class 2 is: {:.4f}, class 3 is: {:.4f}, class 4 is: {:.4f}'.format(
                    metric_dict['accuracy'][0], metric_dict['accuracy'][1], metric_dict[
                        'accuracy'][2], metric_dict['accuracy'][3], metric_dict['accuracy'][4],
                )
                logging.info(msg1)
            else:
                msg = 'epoch: {} phase: {} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1-score: {:.4f}'.format(
                    epoch, phase, epoch_loss, epoch_acc, metric['precision'], metric['recall'], metric['F1_score'])
                print(msg)
                logging.info(msg)
                msg1 = 'The accuracy for class 0 is: {:.4f}, class 1 is: {:.4f}, class 2 is: {:.4f}, class 3 is: {:.4f}, class 4 is: {:.4f}'.format(
                    metric_dict['accuracy'][0], metric_dict['accuracy'][1], metric_dict[
                        'accuracy'][2], metric_dict['accuracy'][3], metric_dict['accuracy'][4],
                )
                logging.info(msg1)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                cnt = 0
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                sklearn_plot(old_stat, prefix)
            elif phase == 'val' and epoch_acc < best_acc:
                cnt += 1
            else:
                continue
        plot_curve(train_acc, dev_acc, 'acc', prefix)
        plot_curve(train_ls, dev_ls, 'ls', prefix)
        if cnt == 10:
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))
            logging.info(
                'Accuracy on dev set has not improved for 10 epochs, stop training early')
            logging.info('Best val Acc: {:4f}'.format(best_acc))
            # load best model weights
            model.load_state_dict(best_model_wts)
            return model

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    logging.info('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


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
