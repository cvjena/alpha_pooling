#!/bin/bash
python prepare-finetuning-batchscript.py \
    --init_weights /path/to/pretrained_models/resnet50/ft_iter_320000.caffemodel \
    --tag cub200 \
    --gpu_id 0 \
    --num_classes 201 \
    --image_root /path/to/cub200/images/ \
    --chop_off_layer last_relu \
    --train_batch_size 8 \
    --architecture resnet50 \
    /path/to/cub200/train_images.txt \
    /path/to/cub200/test_images.txt
 
