import argparse
import os

import numpy as np


def main(data_path_ori: str, data_path_pre: str):
    """
    Args:
        data_path_ori (str): the top directory of the whole original data set
        data_path_pre (str): the top directory of the whole preprocessed data set
    """
    # the name of the output files
    train_txt = 'train.txt'
    val_txt = 'val.txt'
    ori_train_txt = 'ori_train.txt'
    ori_val_txt = 'ori_val.txt'

    # if there are old files, delete them and regenerate them
    if os.path.exists(train_txt):
        print('{} already exists, delete it and regenerate it'.format(train_txt))
        os.remove(train_txt)
    if os.path.exists(val_txt):
        print('{} already exists, delete it and regenerate it'.format(val_txt))
        os.remove(val_txt)

    # the label of each class
    label_dict = {'PM': 0, 'AMD': 1, 'PCV': 2, 'DME': 3, 'NM': 4}

    # explore the top directory of original data set
    for dir in os.listdir(data_path_pre):
        # /home/johnson-lin/work_dir/data/DIP_pre/AMD
        disease_dir = data_path_pre + '/' + dir
        patients_list = []
        for dir1 in os.listdir(disease_dir):
            # /home/johnson-lin/work_dir/data/DIP_pre/AMD/A-5102
            patients_dir = disease_dir + '/' + dir1
            patients_list.append(patients_dir)
        # for each class, randomly choose 80 patients as the train set
        train = np.random.choice(patients_list, 80, replace=False)
        val = []
        # the rest 20 patients in the class are chosen as the test set
        for d in patients_list:
            if d not in train:
                val.append(d)
        # create output files
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
    # create the files for original data set
    train_list = []
    val_list = []
    with open(train_txt) as f:
        for line in f.readlines():
            line_list = line.strip().split('/')
            new_line = data_path_ori + '/' + \
                line_list[-3] + '/' + line_list[-2] + '/' + line_list[-1]
            assert os.path.exists(new_line.split(
                ' ')[0]), 'Error! {}: no such file or directory!'.format(new_line.split(' ')[0])
            train_list.append(new_line + '\n')
    with open(val_txt) as f:
        for line in f.readlines():
            line_list = line.strip().split('/')
            new_line = data_path_ori + '/' + \
                line_list[-3] + '/' + line_list[-2] + '/' + line_list[-1]
            assert os.path.exists(new_line.split(
                ' ')[0]), 'Error! {}: no such file or directory!'.format(new_line.split(' ')[0])
            val_list.append(new_line + '\n')
    with open(ori_train_txt, 'w') as f:
        for line in train_list:
            f.write(line)
    with open(ori_val_txt, 'w') as f:
        for line in val_list:
            f.write(line)


parser = argparse.ArgumentParser(
    description="Divide the whole data set in to train set and test set")
parser.add_argument('--data_path_ori',  type=str,
                    help='the path of the whole original data set, e.g. /home2/wenyang/guest/data/DIP-data')
parser.add_argument('--data_path_pre', type=str,
                    help='the path of the whole preprocessed data set, e.g. /home2/wenyang/guest/data/DIP-data_pre')

if __name__ == '__main__':
    args = parser.parse_args()
    data_path_ori = args.data_path_ori
    data_path_pre = args.data_path_pre
    main(data_path_ori=data_path_ori, data_path_pre=data_path_pre)
