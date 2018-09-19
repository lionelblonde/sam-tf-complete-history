#!/usr/bin/env bash
# Example: ./spawn_jobs_cscs.sh <task> <benchmark> <num_trials> <demos_dir> <num_mpi_workers>
#                               <call_or_not>

cd ../..

python -m imitation.spawners.spawn_jobs_cscs \
    --script_dir="scripts/cscs" \
    --task=$1 \
    --benchmark=$2 \
    --num_trials=$3 \
    --demos_dir=$4 \
    --num_workers=$5 \
    --call=$6
