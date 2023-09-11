#!/bin/bash
#SBATCH -J AD
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks=12
#SBATCH --gres=gpu:1

source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
module unload cudnn8.0-cuda11.1/
module load cudnn7.6-cuda10.1/7.6.5.32
conda activate tf2
cd /mntntfs/diis_data1/zhangmaojun/LCP_paper/

#python3 -u generate_dataset/generate_channel.py
#python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 8 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u single_RB/train_main.py              --Nt 64 --Nr 4 --K 8 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

#python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 9 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u single_RB/train_main.py              --Nt 64 --Nr 4 --K 9 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

#python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u single_RB/train_main.py              --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

#python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 11 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u single_RB/train_main.py              --Nt 64 --Nr 4 --K 11 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

#python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 12 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u single_RB/train_main.py              --Nt 64 --Nr 4 --K 12 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

#python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 13 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u single_RB/train_main.py              --Nt 64 --Nr 4 --K 13 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 14 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u single_RB/train_main.py              --Nt 64 --Nr 4 --K 14 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 15 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u single_RB/train_main.py              --Nt 64 --Nr 4 --K 15 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 16 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u single_RB/train_main.py              --Nt 64 --Nr 4 --K 16 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

