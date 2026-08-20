#!/bin/bash
#SBATCH --job-name=cueq-t-uni
#SBATCH --partition=gpu
#SBATCH --exclude=gpu01,gpu04
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jerry528/mace-ace-pr124/perf_cueq/results/time_uniform_prod.txt
cd /home/jerry528/mace-ace-pr124/perf_cueq
export MAX_ELL=3 MAX_MELL=2 CH=128 BATCH=432 NU=3 CHUNK=250 REP=5 DTYPE=64
export CUEQ_NONSOC_VERBOSE=1
export CUDA_VISIBLE_DEVICES=0        # not gpu01, so device 0 is fine
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUEQ_METHOD=uniform_1d
V=/storage/data/jerry528/cueq_test_011/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$V/cuequivariance_ops/lib:$LD_LIBRARY_PATH"
exec /storage/data/jerry528/cueq_test_011/bin/python scripts/time_integration.py
