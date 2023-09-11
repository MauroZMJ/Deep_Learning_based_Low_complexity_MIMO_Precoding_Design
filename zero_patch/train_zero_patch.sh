#!/bin/bash
#SBATCH -J AD
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1

source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
module unload cudnn8.0-cuda11.1/
module load cudnn7.6-cuda10.1/7.6.5.32
conda activate tf2
cd /mnts2d/diis_data1/zmj/LCP_paper/zero_patch_new/
#python3 -u generate_dataset.py --Nt 64 --Nr 4 --K 12 --dk 2 --B 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

#python3 -u generate_dataset.py --Nt 64 --Nr 4 --K 12 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

python3 -u train.py            --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --B 1 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2

python3 -u train.py            --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --B 1 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2 --data_zp
