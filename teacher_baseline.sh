#!/usr/bin/env bash
python train.py --base resnet50 --sample distance --loss l2_triplet --margin 0.2 --embedding_size 512 --save_dir ../metric_result/teachers/teacher_resnet50_512 > ../metric_result/teachers/teacher_resnet50_512.txt
python train.py --base resnet18 --sample distance --loss l2_triplet --margin 0.2 --embedding_size 128 --save_dir ../metric_result/teachers/resnet18_128 > ../metric_result/teachers/resnet18_128.txt
python train.py --base resnet18 --sample distance --loss l2_triplet --margin 0.2 --embedding_size 64 --save_dir ../metric_result/teachers/resnet18_64 > ../metric_result/teachers/resnet18_64.txt
python train.py --base resnet18 --sample distance --loss l2_triplet --margin 0.2 --embedding_size 32 --save_dir ../metric_result/teachers/resnet18_32 > ../metric_result/teachers/resnet18_32.txt
python train.py --base resnet18 --sample distance --loss l2_triplet --margin 0.2 --embedding_size 16 --save_dir ../metric_result/teachers/resnet18_16 > ../metric_result/teachers/resnet18_16.txt
