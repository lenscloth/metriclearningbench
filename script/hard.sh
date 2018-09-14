#!/usr/bin/env bash

python train.py --epochs 50 --sample hard --save resnet50_cub2011_hard_50.pth > resnet50_cub2011_hard_50.txt
python train.py --epochs 50 --sample semihard --save resnet50_cub2011_semihard_50.pth > resnet50_cub2011_semihard_50.txt
