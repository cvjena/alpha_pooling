# Alpha pooling for fine-grained recognition

## Intro
This repository contains code for our International Conference on Computer Vision publication ``[Generalized Orderless Pooling Performs Implicit Salient Matching](http://openaccess.thecvf.com/content_iccv_2017/html/Simon_Generalized_Orderless_Pooling_ICCV_2017_paper.html)''. It contains scripts for fine-tuning a pre-trained VGG16 model with our presented alpha-pooling approach.

## Abstract of the paper
Most recent CNN architectures use average pooling as a final feature encoding step. In the field of fine-grained recognition, however, recent global representations like bilinear pooling offer improved performance. In this paper, we generalize average and bilinear pooling to "alpha-pooling", allowing for learning the pooling strategy during training. In addition, we present a novel way to visualize decisions made by these approaches. We identify parts of training images having the highest influence on the prediction of a given test image. This allows for justifying decisions to users and also for analyzing the influence of semantic parts. For example, we can show that the higher capacity VGG16 model focuses much more on the bird's head than, e.g., the lower-capacity VGG-M model when recognizing fine-grained bird categories. Both contributions allow us to analyze the difference when moving between average and bilinear pooling. In addition, experiments show that our generalized approach can outperform both across a variety of standard datasets.

## Getting started
You need our custom caffe located at [https://github.com/cvjena/caffe_pp2](https://github.com/cvjena/caffe_pp2), which has our own SignedPowerLayer as well as a [spatial transformer layer](https://github.com/daerduoCarey/SpatialTransformerLayer) for convenience and a [compact bilinear layer](https://github.com/gy20073/compact_bilinear_pooling) for computing the outer product. 
Please compile the python interface as well.
We use python 3 in all our experiments. 

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

The explanation for the usage of the notebook is described in the comments of it.
Please note that we gamma in the scripts refer to alpha in the paper due to last minute renaming of the approach before submission.

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
