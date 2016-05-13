#!/usr/bin/env bash

mkdir -p output
python caffe_train.py >output/stdout 2>output/stderr </dev/null &