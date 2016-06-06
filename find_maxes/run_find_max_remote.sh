#!/usr/bin/env bash
mkdir -p output
#python find_max_acts.py --N 9 --gpu \
#    ../models/caffenet-yos/caffenet-yos-deploy.prototxt ../models/caffenet-yos/caffenet-yos-weights \
#    /home/ubuntu/sdf/images /home/ubuntu/sdf/database_list output/max_out \
#    >output/find_max.out 2>output/find_max.err </dev/null &



#python find_max_acts.py --N 9 --gpu \
#        ../models/my-model/original_deploy.prototxt ../models/my-model/output/sparse_solver.caffemodel \
#    /home/ubuntu/sdg/train /home/ubuntu/sdg/image_list output/max_out_sparse \
#    >output/find_max.out 2>output/find_max.err </dev/null &


python find_max_acts.py --N 9 --gpu \
        ../models/my-model/sigmoid_deploy.prototxt ../models/my-model/output/sigmoid_solver.caffemodel \
    /home/ubuntu/sdg/train /home/ubuntu/sdg/image_list output/max_out_sigmoid \
    >output/find_max.out 2>output/find_max.err </dev/null &