#!/usr/bin/env bash
mkdir -p output
mkdir -p /home/ubuntu/sdg/sigmoid_activations

# python find_activation.py --gpu --num 200000 \
#     ../models/caffenet-yos/caffenet-yos-deploy.prototxt ../models/caffenet-yos/caffenet-yos-weights \
#     /home/ubuntu/sdf/images /home/ubuntu/sdf/database_list activation_out conv3,conv4,conv5,fc6,fc7 \
#     </dev/null >output/stdout 2>output/stderr &


#python find_activation.py --gpu --num 80000 \
#    ../models/my-model/original_deploy.prototxt ../models/my-model/output/sparse_solver.caffemodel \
#    /home/ubuntu/sdg/train /home/ubuntu/sdg/image_list /home/ubuntu/sdg/activations conv3,conv4,conv5,fc6,fc7 \
#    </dev/null >output/stdout 2>output/stderr &

python find_activation.py --gpu --num 80000 \
    ../models/my-model/sigmoid_deploy.prototxt ../models/my-model/output/sigmoid_solver.caffemodel \
    /home/ubuntu/sdg/train /home/ubuntu/sdg/image_list /home/ubuntu/sdg/sigmoid_activations conv3,conv4,conv5,scale6_2,fc7 \
    </dev/null >output/stdout 2>output/stderr &