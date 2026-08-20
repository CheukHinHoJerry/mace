#!/bin/bash
#SBATCH --job-name=cueq-warm
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/home/jerry528/mace-ace-pr124/perf_cueq/results/warm_bench_prod.txt
cd /home/jerry528/mace-ace-pr124/perf_cueq
export MAX_ELL=3 MAX_MELL=2 NU=3 CH=128 BATCH=432 REP=5
exec bash scripts/run_cueq.sh scripts/warm_bench.py
