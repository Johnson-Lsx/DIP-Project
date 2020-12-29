from __future__ import division, print_function

import argparse
import copy
import logging
import os
import time

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import models, transforms
from tqdm import tqdm

from metric import (
    cal_metric, sklearn_cal_metric, sklearn_plot, sklearn_stat, stat)

matplotlib.use('Agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# use PIL Image to read image


def default_loader(path: str):
    """
    Args:
        path (str): the path of the image file.

    Returns:
        img (PIL.Image.Image): the image file.
    """
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

# define your Dataset. Assume each line in your .txt file is [name/one space/label], for example:0001.jpg 1


class customData(Dataset):
    def __init__(self, txt_path, dataset, data_transforms=None, loader=default_loader):
        """
        Args:
            txt_path (str): the path of the text file that contains the paths and labels of image files.
            dataset (str): the value can be 'train' or 'val', it is used to apply transforms on corresponding dataset.
            data_transforms (dict, optional): describes which kind of transform will be applied to the data set. Defaults to None.
            loader (Callable, optional): the function used to fetch a single image of the dataset. Defaults to default_loader.
        """
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [line.strip().split(' ')[0] for line in lines]
            self.img_label = [int(line.strip().split(' ')[-1])
                              for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        label = self.img_label[idx]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label


def plot_curve(train, dev, mode, prefix):
    """AI is creating summary for plot_curve

    Args:
        train (List[float]): the accuracy or loss of the train set
        dev (List[float]): the accuracy or loss of the dev set
        mode (str): plot the accuracy curve or loss curve
        prefix (str): the path to save the image

    Raises:
        ValueError: the mode parameter can only be 'acc' or 'ls'
    """
    if mode == 'acc':
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.plot(train, label='train acc', marker="s")
        plt.plot(dev, label='dev acc', marker="o")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig('./images/' + prefix + '_acc.png')
        plt.close()
    elif mode == 'ls':
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(train, label='train loss', marker="s")
        plt.plot(dev, label='dev loss', marker="o")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig('./images/' + prefix + '_ls.png')
        plt.close()
    else:
        raise ValueError("Unsupported mode type: %s" % mode)


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


def eval(dataloaders, image_datasets, num_class, prefix):
    """
    Args:
        dataloaders (Dict[str: torch.utils.data.DataLoader]): dataloaders that send the data to the model.
        image_datasets (Dict[str: torch.utils.data.Dataset]): training and validation datasets.
        num_class (int): the number of classes.
        prefix (str): the name of image of the confuse matrix
    """
    phase = 'test'
    criterion = nn.CrossEntropyLoss()
    model_path = "./models/" + prefix + "_best_acc.pkl"
    dataset_sizes = {x: len(image_datasets[x]) for x in [phase]}
    model = torch.load(model_path)
    use_auc = True

    for epoch in range(1):
        model.eval()   # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
        old_stat = dict()
        stat_list = list()
        # Iterate over data.
        for inputs, labels in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                loss = criterion(outputs, labels)
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

        if use_auc:
            msg = '{} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1-score: {:.4f} AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, metric['precision'], metric['recall'], metric['F1_score'], metric['auc'])
            print(msg)
            logging.info(msg)
            msg1 = 'The accuracy for class 0 is: {:.4f}, class 1 is: {:.4f}, class 2 is: {:.4f}, class 3 is: {:.4f}, class 4 is: {:.4f}'.format(
                metric_dict['accuracy'][0], metric_dict['accuracy'][1], metric_dict[
                    'accuracy'][2], metric_dict['accuracy'][3], metric_dict['accuracy'][4],
            )
            logging.info(msg1)
        else:
            msg = '{} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1-score: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, metric['precision'], metric['recall'], metric['F1_score'])
            print(msg)
            logging.info(msg)
            msg1 = 'The accuracy for class 0 is: {:.4f}, class 1 is: {:.4f}, class 2 is: {:.4f}, class 3 is: {:.4f}, class 4 is: {:.4f}'.format(
                metric_dict['accuracy'][0], metric_dict['accuracy'][1], metric_dict[
                    'accuracy'][2], metric_dict['accuracy'][3], metric_dict['accuracy'][4],
            )
            logging.info(msg1)
        sklearn_plot(old_stat, prefix + '_eval')


def train(dataloaders, image_datasets, num_class: int, num_epochs: int, model: str, optimizer: str, lr: float, batch_size: int, weight_decay: float = 0, preprocess: str = 'True'):
    """
    Args:
        dataloaders (Dict[str: torch.utils.data.DataLoader]): dataloaders
        image_datasets (Dict[str: torch.utils.data.Dataset]): datasets
        num_class (int): the number of classes.
        num_epochs (int): the number of epochs to train the model.
        model (str): choose which kind of models to use.
        optimizer (str): choose which kind of optimizer to use.
        lr (float): learning rate
        batch_size (int): batch size
        weight_decay (float, optional): weight_decay(L2 penalty). Default is 0.
        preprocess (str): wheather or not to use the preprocessed images
    Return:
        prefix (str): the prefix of the parameter file of the model.
    """
    if model.startswith('resnet'):
        if model == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif model == 'resnet34':
            model_ft = models.resnet34(pretrained=True)
        elif model == 'resnet50':
            model_ft = models.resnet50(pretrained=True)
        else:
            raise ValueError("Unsupported model type: %s" % model)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_class)
        model_ft = model_ft.to(device)
        output_params = list(map(id, model_ft.fc.parameters()))
        feature_params = filter(lambda p: id(
            p) not in output_params, model_ft.parameters())
    elif model.startswith('vgg'):
        if model == 'vgg16':
            model_ft = models.vgg16(pretrained=True)
        elif model == 'vgg19':
            model_ft = models.vgg19(pretrained=True)
        else:
            raise ValueError("Unsupported model type: %s" % model)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_class)
        model_ft = model_ft.to(device)
        output_params = list(map(id, model_ft.classifier[6].parameters()))
        feature_params = filter(lambda p: id(
            p) not in output_params, model_ft.parameters())
    elif model.startswith('Net'):
        model_ft = Net().to(device)
        feature_params = None
    else:
        raise ValueError("Unsupported model type: %s" % model)
    # model_ft = torch.nn.DataParallel(model_ft)#, device_ids=[0,1])
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    assert optimizer in [
        'sgd', 'adam'], 'Not supported optimizer type: {}, only support sgd and adam'.format(optimizer)
    if optimizer == 'sgd':
        if model.startswith('resnet'):
            optimizer_ft = optim.SGD([{'params': feature_params},
                                      {'params': model_ft.fc.parameters(), 'lr': lr * 10}],
                                     lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif model.startswith('vgg'):
            optimizer_ft = optim.SGD([{'params': feature_params},
                                      {'params': model_ft.classifier[6].parameters(), 'lr': lr * 10}],
                                     lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer_ft = optim.SGD(
                model_ft.parameters(), lr=lr, momentum=0.9)
    else:
        if model.startswith('resnet'):
            optimizer_ft = optim.Adam([{'params': feature_params},
                                       {'params': model_ft.fc.parameters(), 'lr': lr * 10}],
                                      lr=lr, weight_decay=weight_decay)
        elif model.startswith('vgg'):
            optimizer_ft = optim.Adam([{'params': feature_params},
                                       {'params': model_ft.classifier[6].parameters(), 'lr': lr * 10}],
                                      lr=lr, weight_decay=weight_decay)
        else:
            optimizer_ft = optim.Adam(
                model_ft.parameters(), lr=lr, weight_decay=weight_decay)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)
    prefix = model + "_bs" + str(batch_size) + "_optim_" + optimizer + "_lr" + str(
        lr) + "_wd" + str(weight_decay) + "_epochs" + str(num_epochs) + "_pre_" + preprocess
    model_ft = train_model(dataloaders, image_datasets, model_ft,
                           criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs, prefix=prefix, num_class=num_class)
    torch.save(model_ft, "./models/" + prefix + "_best_acc.pkl")
    return prefix


def Data_loader(batch_size: int, preprocess: str = 'True'):
    """
    Args:
        batch_size (int): batch size
        preprocess (str): wheather or not to use the preprocessed images
    Returns:
        Tuple[Dict[str: Dataset], Dict[str: DataLoader]]: datasets, dataloaders
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    if preprocess == 'True':
        image_datasets = {x: customData(txt_path=(x + '.txt'),
                                        dataset=x,
                                        data_transforms=data_transforms,
                                        ) for x in ['train', 'val']}
    elif preprocess == 'False':
        image_datasets = {x: customData(txt_path=('ori_' + x + '.txt'),
                                        dataset=x,
                                        data_transforms=data_transforms,
                                        ) for x in ['train', 'val']}
    else:
        raise ValueError("preprocess can only be True or False")
    # wrap your data and label into Tensor
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x])
                     for x in ['train', 'val']}
    print(dataset_sizes)
    return image_datasets, dataloaders


parser = argparse.ArgumentParser(description="Image classification")
parser.add_argument('--batch_size', default=8, type=int,
                    help='batch size, default 8')
parser.add_argument('--num_epochs', default=30, type=int,
                    help='the number of epochs to train the model, default 30')
parser.add_argument('--model', default='resnet18', type=str,
                    help='the model to train, only supported resnet18, 34, 50 and vgg16, 19, default resnet18')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='optimizer to optimize the model parameters, default adam')
parser.add_argument('--lr', default=0.0001, type=float,
                    help='learning rate, default 0.0001')
parser.add_argument('--weight_decay', default=0,
                    type=float, help='weight_decay, default 0')
parser.add_argument('--preprocess', default='True',
                    type=str, help='wheather or not to use the preprocessed images, can only be True or False, default True')

if __name__ == '__main__':
    args = parser.parse_args()
    num_class = 5
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    model = args.model
    optimizer = args.optimizer
    lr = args.lr
    weight_decay = args.weight_decay
    preprocess = args.preprocess
    # check whether pwd has the needed subdirectories, if not make them
    for subdirs in ['images', 'logs', 'models']:
        if not os.path.exists(subdirs):
            os.makedirs(subdirs)

    log_name = './logs/' + model + '_bs' + str(batch_size) + '_optim_' + optimizer + '_lr' + str(
        lr) + '_wd' + str(weight_decay) + '_epochs' + str(num_epochs) + '_pre_' + preprocess + '.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_name,
                        filemode='w')
    logging.info('model: {} batch_size: {} optimizer: {} learning_rate: {:.4f} weight_decay: {:.4f} num_epochs: {} preprocess: {}'.format(
        model, batch_size, optimizer, lr, weight_decay, num_epochs, preprocess))
    image_datasets, dataloaders = Data_loader(
        batch_size=batch_size, preprocess=preprocess)
    prefix = train(dataloaders, image_datasets, num_class,
                   num_epochs, model, optimizer, lr, batch_size, weight_decay, preprocess)
