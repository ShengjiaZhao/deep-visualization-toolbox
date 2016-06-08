#!/usr/bin/env bash
mkdir -p /home/ubuntu/sdg/sigmoid_results

python crop_max_patches.py --N 9 --gpu \
    --do-maxes --do-deconv --do-deconv-norm --do-info \
    output/max_out_sparse ../models/my-model/sigmoid_deploy.prototxt ../models/my-model/output/sigmoid_solver.caffemodel \
    /home/ubuntu/sdg/train /home/ubuntu/sdg/image_list  /home/ubuntu/sdg/sigmoid_results fc6

python crop_max_patches.py --N 9 --gpu \
    --do-maxes --do-deconv --do-deconv-norm --do-info \
    output/max_out_sparse ../models/my-model/sigmoid_deploy.prototxt ../models/my-model/output/sigmoid_solver.caffemodel \
    /home/ubuntu/sdg/train /home/ubuntu/sdg/image_list  /home/ubuntu/sdg/sigmoid_results conv5

python crop_max_patches.py --N 9 --gpu \
    --do-maxes --do-deconv --do-deconv-norm --do-info \
    output/max_out_sparse ../models/my-model/sigmoid_deploy.prototxt ../models/my-model/output/sigmoid_solver.caffemodel \
    /home/ubuntu/sdg/train /home/ubuntu/sdg/image_list /home/ubuntu/sdg/sigmoid_results conv3

python crop_max_patches.py --N 9 --gpu \
    --do-maxes --do-deconv --do-deconv-norm --do-info \
    output/max_out_sparse ../models/my-model/sigmoid_deploy.prototxt ../models/my-model/output/sigmoid_solver.caffemodel \
    /home/ubuntu/sdg/train /home/ubuntu/sdg/image_list  /home/ubuntu/sdg/sigmoid_results conv4
