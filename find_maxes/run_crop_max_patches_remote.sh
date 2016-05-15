#!/usr/bin/env bash
mkdir -p /home/ubuntu/sdg/results

python crop_max_patches.py --N 9 --gpu \
    --do-maxes --do-deconv --do-deconv-norm --do-info \
    output/max_out_sparse ../models/my-model/original_deploy.prototxt ../models/my-model/output/sparse_solver.caffemodel \
    /home/ubuntu/sdf/images /home/ubuntu/sdf/database_list /home/ubuntu/sdg/results conv3

python crop_max_patches.py --N 9 --gpu \
    --do-maxes --do-deconv --do-deconv-norm --do-info \
    output/max_out_sparse ../models/my-model/original_deploy.prototxt ../models/my-model/output/sparse_solver.caffemodel \
    /home/ubuntu/sdg/train /home/ubuntu/sdg/image_list  /home/ubuntu/sdg/results conv4

python crop_max_patches.py --N 9 --gpu \
    --do-maxes --do-deconv --do-deconv-norm --do-info \
    output/max_out_sparse ../models/my-model/original_deploy.prototxt ../models/my-model/output/sparse_solver.caffemodels \
    /home/ubuntu/sdg/train /home/ubuntu/sdg/image_list  /home/ubuntu/sdg/results conv5