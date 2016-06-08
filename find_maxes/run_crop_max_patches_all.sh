#!/usr/bin/env bash
mkdir -p output
./run_crop_max_patches_sigmoid.sh >output/run_crop_max.out 2>output/run_crop_max.err </dev/null &