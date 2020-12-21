import argparse
import os

import numpy as np


def main(data_path_ori: str, data_path_pre: str, use_val: str):
    """
    Args:
        data_path_ori (str): the top directory of the whole original data set
        data_path_pre (str): the top directory of the whole preprocessed data set
        use_val (str): if True, divide the whole data set in to train, validate and test set, else only train and test
    """
    # the name of the output files
    train_txt = 'train.txt'
    val_txt = 'val.txt'
    test_txt = 'test.txt'
    ori_train_txt = 'ori_train.txt'
    ori_val_txt = 'ori_val.txt'
    ori_test_txt = 'ori_test.txt'

    # if there are old files, delete them and regenerate them
    if os.path.exists(train_txt):
        print('{} already exists, delete it and regenerate it'.format(train_txt))
        os.remove(train_txt)
    if os.path.exists(val_txt):
        print('{} already exists, delete it and regenerate it'.format(val_txt))
        os.remove(val_txt)
    if os.path.exists(test_txt):
        print('{} already exists, delete it and regenerate it'.format(test_txt))
        os.remove(test_txt)

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
        test = []
        # the rest 20 patients in the class are chosen as the test set
        for d in patients_list:
            if d not in train:
                test.append(d)
        if use_val == 'True':
            val = np.random.choice(test, 20, replace=False)
            test_tmp = []
            for d in test:
                if d not in val:
                    test_tmp.append(d)
            test = test_tmp
        # create output files
        with open(train_txt, 'a') as f:
            for p in train:
                for file in os.listdir(p):
                    line = str(
                        p) + '/' + str(file) + ' ' + str(label_dict[dir]) + '\n'
                    f.write(line)
        with open(test_txt, 'a') as f:
            for p in test:
                for file in os.listdir(p):
                    line = str(
                        p) + '/' + str(file) + ' ' + str(label_dict[dir]) + '\n'
                    f.write(line)
        if use_val == 'True':
            with open(val_txt, 'a') as f:
                for p in val:
                    for file in os.listdir(p):
                        line = str(
                            p) + '/' + str(file) + ' ' + str(label_dict[dir]) + '\n'
                        f.write(line)
    # create the files for original data set
    train_list = []
    test_list = []
    with open(train_txt) as f:
        for line in f.readlines():
            line_list = line.strip().split('/')
            new_line = data_path_ori + '/' + \
                line_list[-3] + '/' + line_list[-2] + '/' + line_list[-1]
            assert os.path.exists(new_line.split(
                ' ')[0]), 'Error! {}: no such file or directory!'.format(new_line.split(' ')[0])
            train_list.append(new_line + '\n')
    with open(test_txt) as f:
        for line in f.readlines():
            line_list = line.strip().split('/')
            new_line = data_path_ori + '/' + \
                line_list[-3] + '/' + line_list[-2] + '/' + line_list[-1]
            assert os.path.exists(new_line.split(
                ' ')[0]), 'Error! {}: no such file or directory!'.format(new_line.split(' ')[0])
            test_list.append(new_line + '\n')
    with open(ori_train_txt, 'w') as f:
        for line in train_list:
            f.write(line)
    with open(ori_test_txt, 'w') as f:
        for line in test_list:
            f.write(line)
    if use_val == 'True':
        val_list = []
        with open(val_txt) as f:
            for line in f.readlines():
                line_list = line.strip().split('/')
                new_line = data_path_ori + '/' + \
                    line_list[-3] + '/' + line_list[-2] + '/' + line_list[-1]
                assert os.path.exists(new_line.split(
                    ' ')[0]), 'Error! {}: no such file or directory!'.format(new_line.split(' ')[0])
                val_list.append(new_line + '\n')
        with open(ori_val_txt, 'w') as f:
            for line in train_list:
                f.write(line)


parser = argparse.ArgumentParser(
    description="Divide the whole data set in to train set and test set")
parser.add_argument('--data_path_ori',  type=str,
                    help='the path of the whole original data set, e.g. /home2/wenyang/guest/data/DIP_data')
parser.add_argument('--data_path_pre', type=str,
                    help='the path of the whole preprocessed data set, e.g. /home2/wenyang/guest/data/DIP_data_pre')
parser.add_argument('--use_val', type=str,
                    help='if True, divide the whole data set in to train, validate and test set, else only train and test')


if __name__ == '__main__':
    args = parser.parse_args()
    data_path_ori = args.data_path_ori
    data_path_pre = args.data_path_pre
    use_val = args.use_val
    main(data_path_ori=data_path_ori, data_path_pre=data_path_pre, use_val=use_val)
