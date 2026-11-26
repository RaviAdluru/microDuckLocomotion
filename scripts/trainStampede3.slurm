#!/bin/bash
# Trains up to 4 runs at once on one Stampede3 H100 node, one run per GPU.
# Each argument is one run: its name first, then any train.py flags. From the repo root:
#   sbatch scripts/trainStampede3.slurm "hist1Seed1 --historyLength 1 --seed 1" "hist5Seed1 --historyLength 5 --seed 1"
#SBATCH -J duckTrain
#SBATCH -o slurm_%j.out
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A SEE23002
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ra37962@my.utexas.edu

export UV_CACHE_DIR=$SCRATCH/uvCache
mkdir -p runs logs

gpu=0
for spec in "$@"; do
    read -r name flags <<< "$spec"
    echo "GPU $gpu: $name $flags"
    CUDA_VISIBLE_DEVICES=$gpu uv run scripts/train.py --runName $name --logger tensorboard $flags > logs/$name.log 2>&1 &
    gpu=$((gpu + 1))
done
wait
