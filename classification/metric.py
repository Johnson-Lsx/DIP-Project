import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def stat(num_classes, preds, labels, old_stat):
    """
    Args:
        num_classes: the num of classes
        preds: tensor, the predicted classes of the inputs
        labels: tensor, the true classes of the inputs
        old_stat: List[dict], the num of TP, FP, FN of each class
    Return:
        stat: List[dict], the newer version of old_stat
    """
    # 第一次调用函数时，初始化old_stat
    assert len(preds) == len(labels), 'len(preds) != len(labels)'
    if len(old_stat) == 0:
        for i in range(num_classes):
            tmp = dict()
            tmp['TP'] = 0
            tmp['FP'] = 0
            tmp['FN'] = 0
            old_stat.append(tmp)
    for i in range(len(preds)):
        # 预测的和实际的类别相等,该类别的TP数量加1
        if preds[i] == labels[i]:
            old_stat[preds[i]]['TP'] += 1
        else:
            # 两者不相等，对于preds[i]类而言
            # 相当于把不是它的类别预测为它，属于FP
            # 对于labels[i]类而言，相当于把实际
            # 是它的类别预测为不是它，属于FN
            old_stat[preds[i]]['FP'] += 1
            old_stat[labels[i]]['FN'] += 1
    return old_stat


def cal_metric(stat):
    precision = list()
    recall = list()
    F1_score = list()
    metric = dict()
    for i in range(len(stat)):
        p = float(stat[i]['TP'])/(stat[i]['TP']+stat[i]['FP'])
        r = float(stat[i]['TP'])/(stat[i]['TP']+stat[i]['FN'])
        f = 2/(1/p + 1/r)
        precision.append(p)
        recall.append(r)
        F1_score.append(f)
    metric['precision'] = np.mean(precision)
    metric['recall'] = np.mean(recall)
    metric['F1_score'] = np.mean(F1_score)
    return metric


def sklearn_stat(outputs, labels, old_stat, use_auc):
    """
    Args:
        outputs: tensor, the output of the model
        labels: tensor, the true classes of the inputs
        old_stat: Dict[str, list[int]], the history of preds and labels
        use_auc: bool, whether to compute the AUC metric
    Return:
        stat: Dict[str, list[int]], the newer version of old_stat
    """
    # 第一次调用函数时，初始化old_stat
    outputs = outputs.cpu()
    preds = outputs.argmax(dim=1)
    net = nn.Softmax(dim=1)
    net.to(outputs.device)
    prob = net(outputs)
    prob = prob.detach().numpy()
    assert len(preds) == len(labels), 'len(preds) != len(labels)'
    assert prob.shape == outputs.shape, 'prob.shape != outputs.shape'
    if len(old_stat) == 0:
        old_stat['preds'] = list()
        old_stat['labels'] = list()
        if use_auc:
            old_stat['scores'] = list()

    for i in range(len(preds)):
        old_stat['preds'].append(preds[i].item())
        old_stat['labels'].append(labels[i].item())
    
    if use_auc:
        if len(old_stat['scores']) == 0:
            old_stat['scores'].append(prob)
        else:
            tmp = old_stat['scores'][0]
            del old_stat['scores'][0]
            assert len(old_stat['scores']) == 0, 'len(old_stat[scores]) != 0'
            old_stat['scores'].append(np.vstack((tmp, prob)))
    return old_stat


def sklearn_cal_metric(stat, use_auc):
    metric = dict()
    metric['precision'] = precision_score(
        stat['labels'], stat['preds'], average='macro')
    metric['recall'] = recall_score(
        stat['labels'], stat['preds'], average='macro')
    metric['F1_score'] = f1_score(
        stat['labels'], stat['preds'], average='macro')
    if use_auc:
        metric['auc'] = roc_auc_score(
            stat['labels'], stat['scores'][0], multi_class='ovr')
    return metric
