#!/bin/bash
export CUDA_VISIBLE_DEVICES=1     # gpu01 device 0 is faulty
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
exec /home/jerry528/miniconda3/envs/mace-mlmm2/bin/python /storage/data/jerry528/nonsoc_repro/bench/gpu_bench.py
