## README

### 项目简介

1. 本项目用于对眼部疾病的OCT图像进行分类，目前支持ResNet18, 34, 50和VGG16，19这5个经典的网络，在测试集上的准确率可以达到90%以上。另外，我们也尝试了使用3D-ResNet对图像进行分类，由于3D-ResNet参数量巨大，需要海量数据才能训练好，所以，目前3D-ResNet的准确率并没有超过经典网络。

2. 本项目基于`pytorch1.6`深度学习框架，运行过程也依赖如下一些`python`库：

   + `matplotlib`
   + `seaborn`
   + `PIL`
   + `torchvision`
   + `opencv-python`
   + `sklearn`
   + `tqdm`

   运行时如果仍然提示某些库未安装，请根据报错信息安装相应的库。

3. 本项目包含以下`python`文件：

   ```
   metric.py: 用于分类任务常见指标的计算
   
   preprocess.py: 用于经典网络的数据预处理
   divide_trdev.py: 用于经典网络的数据集划分
   classification.py: 用于经典网络的训练
   
   dataset.py: 用于3D-ResNet网络的数据集划分、数据加载
   resnet.py: 用于3D-ResNet网络的实现
   train_3d.py: 用于3D-ResNet网络的训练
   ```

   运行本项目时，请确保上述文件均位于同级目录之下。

4. 本项目的数据集包含5类眼部疾病，每类疾病有100名病人，每名病人大约有19张OCT图像，数据集的目录结构大致如下：

   ```
   DIP_data
   |__AMD
   |   |__A-0001
   |	|	   |_Z-0000.tif
   |   |      |_Z-0001.tif
   |   |      |_ ... 
   |   |      |_Z-0018.tif
   |   |__A-0002
   |   |      |_...
   |   |__...
   |__DME
   |__NM
   |__PCV
   |__PM
   ```

------

### 经典网络训练流程

1. 运行预处理脚本`preprocess.py`生成预处理后的数据集，该脚本的使用说明如下：

   ```
   usage: preprocess.py [-h] [--src_root_path SRC_ROOT_PATH]
                     [--dst_root_path DST_ROOT_PATH]

   Data preprocess for image classification

   optional arguments:
   -h, --help            show this help message and exit
   --src_root_path SRC_ROOT_PATH
                        the absolute path of the whole dataset
   --dst_root_path DST_ROOT_PATH
                        the path to store the preprocessed image
   ```

2. 预处理脚本会将处理之后的所有图片按照原来的目录结构保存，所以会有两份数据，一份是原始数据，一份是预处理之后的数据，假设原始数据的顶层目录路径为`/your/path/to/DIP_data`，预处理之后数据的顶层目录路径为`/your/path/to/DIP_data_pre`。

3. 运行`divide_trdev.py`，划分数据集。该脚本的使用说明如下：

   ```
   python divide_trdev.py --help
   usage: divide_trdev.py [-h] [--data_path_ori DATA_PATH_ORI]
                          [--data_path_pre DATA_PATH_PRE] [--use_val USE_VAL]
   
   Divide the whole data set in to train set and test set
   
   optional arguments:
     -h, --help            show this help message and exit
     --data_path_ori DATA_PATH_ORI
                           the path of the whole original data set, e.g.
                           /home2/wenyang/guest/data/DIP_data
     --data_path_pre DATA_PATH_PRE
                           the path of the whole preprocessed data set, e.g.
                           /home2/wenyang/guest/data/DIP_data_pre
     --use_val USE_VAL     if True, divide the whole data set in to train,
                           validate and test set, else only train and test
   ```

   比如下面的命令就会将数据集划分为训练集和测试集，并生成对应的路径文件：

   ```bash
   python divide_trdev.py --data_path_ori /home2/wenyang/guest/data/DIP_data --data_path_pre DATA_PATH_PRE \
    --use_val False
   ```

   运行之后，当前目录下将新增`train.txt`、`val.txt`、`ori_train.txt`、`ori_val.txt`。

4. 运行`classification.py`，进行网络训练。该脚本的使用说明如下：

   ```
   python classification.py -h  
   usage: classification.py [-h] [--batch_size BATCH_SIZE]
                            [--num_epochs NUM_EPOCHS] [--model MODEL]
                            [--optimizer OPTIMIZER] [--lr LR]
                            [--weight_decay WEIGHT_DECAY]
                            [--preprocess PREPROCESS]
   
   Image classification
   
   optional arguments:
     -h, --help            show this help message and exit
     --batch_size BATCH_SIZE
                           batch size, default 8
     --num_epochs NUM_EPOCHS
                           the number of epochs to train the model, default 25
     --model MODEL         the model to train, only supported resnet18, 34, 50
                           and vgg16, 19, default resnet18
     --optimizer OPTIMIZER
                           optimizer to optimize the model parameters, default
                           adam
     --lr LR               learning rate, default 0.0001
     --weight_decay WEIGHT_DECAY
                           weight_decay, default 0
     --preprocess PREPROCESS
                           wheather or not to use the preprocessed images, can
                           only be True or False, default True
   ```

   该脚本所有需要的参数已经提供了默认值，直接运行`python classification.py`，相当于运行下面的命令：

   ```bash
    python classification.py --model resnet18 --optimizer adam --batch_size 8 --lr 0.0001 \
                             --weight_decay 0 --preprocess False --num_epochs 30
   ```

    表示使用的模型为ResNet18,，优化器为Adam，batch size为8，学习率为0.0001，权重衰减为0，图像不进行预处理，训练轮数为30轮。

   首次运行该脚本会在当前目录下生成`images/`、`logs/`、`models/`三个子目录，三个子目录中存放的内容如下：

   ```
   pwd
   |__images/		模型在训练集和测试集上的loss曲线，准确率曲线
   |__logs/		模型训练过程每一轮在训练集和测试集上的准确率等指标
   |__models/		在测试集上准确率最高的模型的参数文件
   ```

------

### 3D-ResNet网络的训练流程

1. 运行`train_3d.py`文件即可进行训练，该脚本的使用说明如下：

   ```
   usage: train_3d.py [-h] [--img_root_dir IMG_ROOT_DIR]
                      [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
                      [--model_depth MODEL_DEPTH] [--optimizer OPTIMIZER]
                      [--lr LR] [--weight_decay WEIGHT_DECAY]
   
   Image classification using 3D ResNet
   
   optional arguments:
     -h, --help            show this help message and exit
     --img_root_dir IMG_ROOT_DIR
                           the absolute path of the top directory of the whole
                           dataset
     --batch_size BATCH_SIZE
                           batch size, default 4
     --num_epochs NUM_EPOCHS
                           the number of epochs to train the model, default 30
     --model_depth MODEL_DEPTH
                           the depth of the ResNet, only support [10, 18, 34, 50,
                           101, 152, 200], default 10
     --optimizer OPTIMIZER
                           optimizer to optimize the model parameters, default
                           adam
     --lr LR               learning rate, default 0.001
     --weight_decay WEIGHT_DECAY
                           weight_decay, default 0
   ```

   其中必选参数只有`--img_root_dir`，改参数代表的是整个数据集的顶层目录。比如下面的命令：

   ```bash
    python train_3d.py --img_root_dir /home2/wenyang/guest/data/DIP_data --batch_size 2 --model_depth 10
   ```

   就表示利用位于`/home2/wenyang/guest/data/DIP_data`的数据集，训练深度为10的3D-ResNet，且batch size大小取2。

2. 训练的相关结果文件也会保存在当前目录的`images/`、`logs/`、`models/`三个子目录中。

