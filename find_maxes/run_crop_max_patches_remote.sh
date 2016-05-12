#!/usr/bin/env bash
python crop_max_patches.py --N 9 --gpu \
    --do-maxes --do-deconv --do-deconv-norm --do-info \
    ../max_out ../models/caffenet-yos/caffenet-yos-deploy.prototxt ../models/caffenet-yos/caffenet-yos-weights \
    /home/ubuntu/sdf/images /home/ubuntu/sdf/database_list /home/ubuntu/sdf/results fc6

python crop_max_patches.py --N 9 --gpu \
    --do-maxes --do-deconv --do-deconv-norm --do-info \
    ../max_out ../models/caffenet-yos/caffenet-yos-deploy.prototxt ../models/caffenet-yos/caffenet-yos-weights \
    /home/ubuntu/sdf/images /home/ubuntu/sdf/database_list /home/ubuntu/sdf/results fc7

python crop_max_patches.py --N 9 --gpu \
    --do-maxes --do-deconv --do-deconv-norm --do-info \
    ../max_out ../models/caffenet-yos/caffenet-yos-deploy.prototxt ../models/caffenet-yos/caffenet-yos-weights \
    /home/ubuntu/sdf/images /home/ubuntu/sdf/database_list /home/ubuntu/sdf/results fc8