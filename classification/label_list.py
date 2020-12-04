import os
import glob

# 分三级目录，如A/B/a.jpg
# input_path为一级目录；
#
def creat_filelist(input_path, classes):
    # 创建三级目录
    # index 一定是str类型，不可以为int
    dir_image1 = []  # 二级目录
    file_list = []  # 三级目录
    for index, name in enumerate(classes):
        #print('index', index)
        index_str = str(index)
        #dir_image1_temp = input_path + '/' + name + '/'
        dir_image1_temp = input_path + name
        print(dir_image1_temp)  # /home2/wenyang/research/classfication/dataset/train/PM

        for dir2 in os.listdir(dir_image1_temp):
            for image_index in os.listdir(dir_image1_temp+'/'+dir2):
                dir_image2_temp = dir_image1_temp + '/' + dir2 + '/' + image_index + ' ' + index_str
                # dir_image2_temp1 = dir_image2_temp.join(' ')
                # dir_image2_temp2 = dir_image2_temp.join(index)
                file_list.append(dir_image2_temp)
        '''
        for dir2 in os.listdir(dir_image1_temp):
            dir_image2_temp = dir_image1_temp + '/' + dir2 + ' ' + index_str
            # dir_image2_temp1 = dir_image2_temp.join(' ')
            # dir_image2_temp2 = dir_image2_temp.join(index)
            file_list.append(dir_image2_temp)
        '''
    return dir_image1, file_list


def creat_txtfile(output_path, file_list):
    with open(output_path, 'w') as f:
        for list in file_list:
            print(list)
            f.write(str(list) + '\n')


def main():
    train_dir = '/home2/wenyang/research/classfication/dataset/train/'
    test_dir = '/home2/wenyang/research/classfication/dataset/val/'
    dir_image1 = os.listdir(train_dir)
    classes = dir_image1  # ['PM', 'AMD', 'PCV', 'DME', 'NM']
    # print(classes)
    dir_list, file_list = creat_filelist(train_dir, classes)
    # print(file_list[0:3])
    train_path = './train_our.txt'
    test_path = './test_our.txt'
    creat_txtfile(train_path, file_list)
    list = glob.glob(r"/home2/wenyang/research/classfication/dataset/train/*/*/*.tif")
    print("list", len(list))

if __name__ == '__main__':
    main()