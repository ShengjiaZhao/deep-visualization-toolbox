#!/usr/bin/env bash
mkdir -p output
python find_max_acts.py --N 9 --gpu \
    ../models/caffenet-yos/caffenet-yos-deploy.prototxt ../models/caffenet-yos/caffenet-yos-weights \
    /home/ubuntu/sdf/images /home/ubuntu/sdf/database_list output/max_out \
    >output/find_max.out 2>output/find_max.err </dev/null &
