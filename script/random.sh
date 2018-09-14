#!/usr/bin/env bash

python train.py --epochs 50 --sample distance --save resnet50_cub2011_distance_50.pth > resnet50_cub2011_distance_50.txt
python train.py --epochs 50 --sample random --save resnet50_cub2011_random_50.pth > resnet50_cub2011_random_50.txt
