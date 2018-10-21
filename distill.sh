#!/usr/bin/env bash

# Distance and Angle
python train_distill.py --base resnet18 --dist_ratio 1 --angle_ratio 0.5 --no_normalize --embedding_size 128 --teacher_embedding_size 512 --teacher_load ../metric_result/triplet/teacher_resnet50_512/best.pth --save_dir ../metric_result/distill_distangle/distangle_distill_resnet18_128 > ../metric_result/distill_distangle/distangle_distill_resnet18_128.txt
python train_distill.py --base resnet18 --dist_ratio 1 --angle_ratio 0.5 --no_normalize --embedding_size 64 --teacher_embedding_size 512 --teacher_load ../metric_result/triplet/teacher_resnet50_512/best.pth --save_dir ../metric_result/distill_distangle/distangle_distill_resnet18_64 > ../metric_result/distill_distangle/distangle_distill_resnet18_64.txt
python train_distill.py --base resnet18 --dist_ratio 1 --angle_ratio 0.5 --no_normalize --embedding_size 32 --teacher_embedding_size 512 --teacher_load ../metric_result/triplet/teacher_resnet50_512/best.pth --save_dir ../metric_result/distill_distangle/distangle_distill_resnet18_32 > ../metric_result/distill_distangle/distangle_distill_resnet18_32.txt
python train_distill.py --base resnet18 --dist_ratio 1 --angle_ratio 0.5 --no_normalize --embedding_size 16 --teacher_embedding_size 512 --teacher_load ../metric_result/triplet/teacher_resnet50_512/best.pth --save_dir ../metric_result/distill_distangle/distangle_distill_resnet18_16 > ../metric_result/distill_distangle/distangle_distill_resnet18_16.txt

# Distance and Angle Normalized
python train_distill.py --base resnet18 --dist_ratio 1 --angle_ratio 0.5 --embedding_size 128 --teacher_embedding_size 512 --teacher_load ../metric_result/triplet/teacher_resnet50_512/best.pth --save_dir ../metric_result/distill_distangle/l2_distangle_distill_resnet18_128 > ../metric_result/distill_distangle/l2_distangle_distill_resnet18_128.txt
python train_distill.py --base resnet18 --dist_ratio 1 --angle_ratio 0.5 --embedding_size 64 --teacher_embedding_size 512 --teacher_load ../metric_result/triplet/teacher_resnet50_512/best.pth --save_dir ../metric_result/distill_distangle/l2_distangle_distill_resnet18_64 > ../metric_result/distill_distangle/l2_distangle_distill_resnet18_64.txt
python train_distill.py --base resnet18 --dist_ratio 1 --angle_ratio 0.5 --embedding_size 32 --teacher_embedding_size 512 --teacher_load ../metric_result/triplet/teacher_resnet50_512/best.pth --save_dir ../metric_result/distill_distangle/l2_distangle_distill_resnet18_32 > ../metric_result/distill_distangle/l2_distangle_distill_resnet18_32.txt
python train_distill.py --base resnet18 --dist_ratio 1 --angle_ratio 0.5 --embedding_size 16 --teacher_embedding_size 512 --teacher_load ../metric_result/triplet/teacher_resnet50_512/best.pth --save_dir ../metric_result/distill_distangle/l2_distangle_distill_resnet18_16 > ../metric_result/distill_distangle/l2_distangle_distill_resnet18_16.txt
