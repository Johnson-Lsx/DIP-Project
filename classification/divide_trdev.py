import argparse
import os

import numpy as np


def main(data_path: str, pre: bool = True):
    if pre:
        train_txt = 'train.txt'
        val_txt = 'val.txt'
    else:
        train_txt = 'ori_train.txt'
        val_txt = 'ori_val.txt'
    if os.path.exists(train_txt):
        print('{} already exists, delete it and regenerate it'.format(train_txt))
        os.remove(train_txt)
    if os.path.exists(val_txt):
        print('{} already exists, delete it and regenerate it'.format(val_txt))
        os.remove(val_txt)

    label_dict = {'PM': 0, 'AMD': 1, 'PCV': 2, 'DME': 3, 'NM': 4}
    for dir in os.listdir(data_path):
        dir1 = data_path + '/' + dir  # /home/johnson-lin/work_dir/data/DIP_pre/AMD
        tmp = []
        for dir2 in os.listdir(dir1):
            dir3 = dir1 + '/' + dir2  # /home/johnson-lin/work_dir/data/DIP_pre/AMD/A-5102
            tmp.append(dir3)
        train = np.random.choice(tmp, 80, replace=False)
        val = []
        for d in tmp:
            if d not in train:
                val.append(d)
        with open(train_txt, 'a') as f:
            for p in train:
                for file in os.listdir(p):
                    line = str(
                        p) + '/' + str(file) + ' ' + str(label_dict[dir]) + '\n'
                    f.write(line)
        with open(val_txt, 'a') as f:
            for p in val:
                for file in os.listdir(p):
                    line = str(
                        p) + '/' + str(file) + ' ' + str(label_dict[dir]) + '\n'
                    f.write(line)


parser = argparse.ArgumentParser(
    description="Divide the whole data set in to train set and test set")
parser.add_argument('--data_path',  type=str,
                    help='the path of the whole data set, e.g. /home2/wenyang/guest/data')
parser.add_argument('--pre', default='True', type=str,
                    help='to generate the txt file of original dataset or of the preprocessed dataset')

if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.data_path
    pre = args.pre
    if pre == 'True':
        main(data_path=data_path, pre=True)
    elif pre == 'False':
        main(data_path=data_path, pre=False)
    else:
        raise ValueError("Unsupported type: %s" % pre)

