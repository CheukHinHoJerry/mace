#!/bin/bash
#SBATCH --job-name=cueq-scale
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=8:00:00
#SBATCH --output=/home/jerry528/mace-ace-pr124/perf_cueq/results/scaling_uniform.txt
cd /home/jerry528/mace-ace-pr124/perf_cueq
export CUDA_VISIBLE_DEVICES=1        # gpu01 device 0 is faulty
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUEQ_METHOD=uniform_1d CUEQ_NONSOC_VERBOSE=1
V=/storage/data/jerry528/cueq_test_011/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$V/cuequivariance_ops/lib:$LD_LIBRARY_PATH"
PY=/storage/data/jerry528/cueq_test_011/bin/python
# Grow the path count deliberately: m_ell drives the merged-Q width, max_ell the spatial nnz.
for cfg in "2 1" "3 1" "2 2"; do
  set -- $cfg
  echo "############ max_ell=$1 m_ell=$2 ############"
  MAX_ELL=$1 MAX_MELL=$2 CH=128 BATCH=432 NU=3 CHUNK=250 REP=5 DTYPE=64 \
    timeout 7200 $PY scripts/time_integration.py
  echo "############ exit=$? ############"
done
