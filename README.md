# Alpha pooling for fine-grained recognition
This repository contains code for our International Conference on Computer Vision publication ``[Generalized Orderless Pooling Performs Implicit Salient Matching](http://openaccess.thecvf.com/content_iccv_2017/html/Simon_Generalized_Orderless_Pooling_ICCV_2017_paper.html)''. It contains scripts for fine-tuning a pre-trained VGG16 model with our presented alpha-pooling approach.

## Abstract of the paper
Most recent CNN architectures use average pooling as a final feature encoding step. In the field of fine-grained recognition, however, recent global representations like bilinear pooling offer improved performance. In this paper, we generalize average and bilinear pooling to "alpha-pooling", allowing for learning the pooling strategy during training. In addition, we present a novel way to visualize decisions made by these approaches. We identify parts of training images having the highest influence on the prediction of a given test image. This allows for justifying decisions to users and also for analyzing the influence of semantic parts. For example, we can show that the higher capacity VGG16 model focuses much more on the bird's head than, e.g., the lower-capacity VGG-M model when recognizing fine-grained bird categories. Both contributions allow us to analyze the difference when moving between average and bilinear pooling. In addition, experiments show that our generalized approach can outperform both across a variety of standard datasets.

## Getting started
You need our custom caffe located at [https://github.com/cvjena/caffe_pp2](https://github.com/cvjena/caffe_pp2), which has our own SignedPowerLayer with learnable power as well as a [spatial transformer layer](https://github.com/daerduoCarey/SpatialTransformerLayer) used for on-the-fly image resizing and a [compact bilinear layer](https://github.com/gy20073/compact_bilinear_pooling) for computing the outer product in an efficient manner. Please clone and compile caffe_pp2 as well as its python interface. We use python 3 in all our experiments. 

## Preparation of the dataset
We use an ImageData layer in our experiments. This layer is required in order to use the scripts provided here. Hence you will need a list of train images and a list of test images. Each file should contain the path to the respective images relative to `--image_root` and the label as integer separated by comma. This means, the files should look like

```
/path/to/dataset/class1/image1.jpg 1
/path/to/dataset/class1/image2.jpg 1
/path/to/dataset/class2/image1.jpg 2
/path/to/dataset/class2/image2.jpg 2
```

The path to these files is used in the following scripts and are called *train_imagelist* and *val_imagelist*.

## How to learn an alpha-pooling model
We provide a batch script and an Jupyter notebook to prepare the fine-tuning. 
The usage of the batch script is described in the --help message:

    usage: prepare-finetuning-batchscript.py [-h] [--init_weights INIT_WEIGHTS]
                                            [--label LABEL] [--gpu_id GPU_ID]
                                            [--num_classes NUM_CLASSES]
                                            [--image_root IMAGE_ROOT]
                                            train_imagelist val_imagelist

    Prepare fine-tuning of multiscale alpha pooling. The working directory should
    contain train_val.prototxt of vgg16. The models will be created in the
    subfolders.

    positional arguments:
    train_imagelist       Path to imagelist containing the training images. Each
                            line should contain the path to an image followed by a
                            space and the class ID.
    val_imagelist         Path to imagelist containing the validation images.
                            Each line should contain the path to an image followed
                            by a space and the class ID.

    optional arguments:
    -h, --help            show this help message and exit
    --init_weights INIT_WEIGHTS
                            Path to the pre-trained vgg16 model
    --label LABEL         Label of the created output folder
    --gpu_id GPU_ID       ID of the GPU to use
    --num_classes NUM_CLASSES
                            Number of object categories
    --image_root IMAGE_ROOT
                            Image root folder, used to set the root_folder
                            parameter of the ImageData layer of caffe.

The explanation for the usage of the notebook is described in the comments of it. Please note that gamma in the scripts refers to alpha in the paper due to last minute renaming of the approach before submission. 

The script preprare the prototxt and solver for learning the model. In addition, they also learn the last classification layer already. After the preparation, you can fine-tune the network using the created ft.solver file in the finetuning subfolder. *Please note that our implementation only supports GPU computation, as the SignedPowerLayer in caffe_pp2 has only a GPU implementation at the moment.*


## How to learn another architecture 
The code shows the fine-tuning preparation for VGG16. If you want to learn another model, you will need a train_val.prototxt, which has two ImageData layers. It is probably the best to take your existing train_val.prototxt and replace your data layers with the ImageData layers of our VGG16 train_val.prototxt. Our script does not support LMDB or any other types of layers, but could be probably adapted for it. After these adjustments, you might also need to adjust the notebook or prepare-finetuning-batchscript.py, depending on what you are using. 

Feel free to try any other model, for example our caffe implementation of ResNet50 from https://github.com/cvjena/cnn-models/tree/master/ResNet_preact/ResNet50_cvgj

## Accuracy 
With VGG16 and a resolution of 224 and 560 pixels on the smaller side of the image, you should achieve the 85.3% top-1 accuracy reported in the paper. Complete list of results:

|Dataset|CUB200-2011|Aircraft|40 actions-|
|---|---|---|---|
|classes / images| 200 / 12k | 89 / 10k |40 / 9.5k|
Previous| 81.0% [24]| 72.5% [6]| 72.0% [36]|
||82.0% [17]| 78.0% [22] |80.9% [4]|
||84.5% [34] |80.7% [13]| 81.7% [22]|
|Special case: bilinear [19] |84.1%| 84.1% |-|
|Learned strategy (Ours)| 85.3% |85.5% |86.0%|

Note: running the training longer the the predefined number of itertions leads to a higher accuracy and is necessary to reproduce the paper results. 

## Citation
Please cite the corresponding ICCV 2017 publication if our models helped your research:

```
@inproceedings{Simon17_GOP,
title = {Generalized orderless pooling performs implicit salient matching},
booktitle = {International Conference on Computer Vision (ICCV)},
author = {Marcel Simon and Yang Gao and Trevor Darrell and Joachim Denzler and Erik Rodner},
year = {2017},
}
```

### License and support
The code is released under BSD 2-clause license allowing both academic and commercial use. I would appreciate if you give credit to this work by citing our paper in academic works and referencing to this Github repository in commercial works. If you need any support, please open an issue or contact [Marcel Simon](https://marcelsimon.com/).
