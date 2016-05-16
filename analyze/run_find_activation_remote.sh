#!/usr/bin/env bash
mkdir -p output
mkdir -p /home/ubuntu/sdg/activations

# python find_activation.py --gpu --num 200000 \
#     ../models/caffenet-yos/caffenet-yos-deploy.prototxt ../models/caffenet-yos/caffenet-yos-weights \
#     /home/ubuntu/sdf/images /home/ubuntu/sdf/database_list activation_out conv3,conv4,conv5,fc6,fc7 \
#     </dev/null >output/activation.out 2>output/activation.err &


python find_activation.py --gpu --num 5000 \
    ../models/my-model/original_deploy.prototxt ../models/my-model/output/sparse_solver.caffemodel \
    /home/ubuntu/sdg/train /home/ubuntu/sdg/image_list /home/ubuntu/sdg/activations conv3,conv4,conv5,fc6,fc7 \
    </dev/null >output/activation.out 2>output/activation.err &