#!/usr/bin/env bash
python train_distill.py --base resnet18 --base_teacher resnet50 --angle_ratio 1 --embedding_size 64 --teacher_embedding_size 512 --teacher_load resnet50_512_sgd/best.pth --no_normalize --save_dir ../metric_result/2018.10.14/resnet18_64_fc_angle1_try1 > ../metric_result/2018.10.14/resnet18_64_fc_angle1_try1.txt
python train_distill.py --base resnet18 --base_teacher resnet50 --angle_ratio 1 --embedding_size 64 --teacher_embedding_size 512 --teacher_load resnet50_512_sgd/best.pth --no_normalize --save_dir ../metric_result/2018.10.14/resnet18_64_fc_angle1_try2 > ../metric_result/2018.10.14/resnet18_64_fc_angle1_try2.txt
python train_distill.py --base resnet18 --base_teacher resnet50 --angle_ratio 1 --embedding_size 64 --teacher_embedding_size 512 --teacher_load resnet50_512_sgd/best.pth --no_normalize --save_dir ../metric_result/2018.10.14/resnet18_64_fc_angle1_try3 > ../metric_result/2018.10.14/resnet18_64_fc_angle1_try3.txt
python train_distill.py --base resnet18 --base_teacher resnet50 --angle_ratio 1 --embedding_size 64 --teacher_embedding_size 512 --teacher_load resnet50_512_sgd/best.pth --no_normalize --save_dir ../metric_result/2018.10.14/resnet18_64_fc_angle1_try4 > ../metric_result/2018.10.14/resnet18_64_fc_angle1_try4.txt
python train_distill.py --base resnet18 --base_teacher resnet50 --angle_ratio 1 --embedding_size 64 --teacher_embedding_size 512 --teacher_load resnet50_512_sgd/best.pth --no_normalize --save_dir ../metric_result/2018.10.14/resnet18_64_fc_angle1_try5 > ../metric_result/2018.10.14/resnet18_64_fc_angle1_try5.txt


