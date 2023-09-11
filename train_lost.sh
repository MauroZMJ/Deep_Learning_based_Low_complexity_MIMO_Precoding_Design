#!/bin/bash
#SBATCH -J AD
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1

source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
module unload cudnn8.0-cuda11.1/
module load cudnn7.6-cuda10.1/7.6.5.32
conda activate tf2
cd /mntntfs/diis_data1/zhangmaojun/LCP_paper/

#python3 -u learn_UW/train_main.py       --Nt 64 --Nr 4 --K 11 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/train_main.py       --Nt 16 --Nr 2 --K 4 --dk 1 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/train_main.py       --Nt 16 --Nr 2 --K 5 --dk 1 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/train_main.py       --Nt 16 --Nr 2 --K 6 --dk 1 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/train_main.py       --Nt 16 --Nr 2 --K 7 --dk 1 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/train_main.py       --Nt 16 --Nr 2 --K 8 --dk 1 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
