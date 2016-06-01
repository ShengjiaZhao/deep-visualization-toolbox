#!/usr/bin/env bash

mkdir -p output
python caffe_train_sigmoid.py >output/stdout 2>output/stderr </dev/null &