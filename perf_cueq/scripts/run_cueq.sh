#!/bin/bash
export CUDA_VISIBLE_DEVICES=1                      # gpu01 device 0 is faulty
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
V=/storage/data/jerry528/cueq_test/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$V/cuequivariance_ops/lib:$LD_LIBRARY_PATH"
exec /storage/data/jerry528/cueq_test/bin/python "$@"
