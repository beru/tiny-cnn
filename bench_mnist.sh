#!/usr/bin/env bash
NOW=$(date +"%Y%m%d_%H%M%S")
FILENAME=./bench_mnist_result_${NOW}.txt
time /usr/bin/time -pao ${FILENAME} ./bin/Release/example_mnist_train.exe ./data/ | tee ${FILENAME}
