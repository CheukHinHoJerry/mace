#!/bin/bash
# cueq 0.11.1, pure-PyTorch "naive" method: isolates the ENCODING/math from the CUDA kernel.
export CUDA_VISIBLE_DEVICES=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUEQ_METHOD=naive
V=/storage/data/jerry528/cueq_test_011/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$V/cuequivariance_ops/lib:$LD_LIBRARY_PATH"
exec /storage/data/jerry528/cueq_test_011/bin/python "$@"
