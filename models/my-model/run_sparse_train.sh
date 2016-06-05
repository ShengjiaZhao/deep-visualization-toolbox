#!/usr/bin/env bash

mkdir -p output
python caffe_train_sparse.py sparse_solver.prototxt >output/stdout 2>output/stderr </dev/null &