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

python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u multi_RB/train_main.py               --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/generate_dataset.py         --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/train_main.py               --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 2 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u multi_RB/train_main.py               --Nt 64 --Nr 4 --K 10 --dk 2 --B 2 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/generate_dataset.py         --Nt 64 --Nr 4 --K 10 --dk 2 --B 2 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/train_main.py               --Nt 64 --Nr 4 --K 10 --dk 2 --B 2 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 3 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u multi_RB/train_main.py               --Nt 64 --Nr 4 --K 10 --dk 2 --B 3 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/generate_dataset.py         --Nt 64 --Nr 4 --K 10 --dk 2 --B 3 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/train_main.py               --Nt 64 --Nr 4 --K 10 --dk 2 --B 3 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u multi_RB/train_main.py               --Nt 64 --Nr 4 --K 10 --dk 2 --B 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/generate_dataset.py         --Nt 64 --Nr 4 --K 10 --dk 2 --B 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/train_main.py               --Nt 64 --Nr 4 --K 10 --dk 2 --B 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

python3 -u generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 5 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u multi_RB/train_main.py               --Nt 64 --Nr 4 --K 10 --dk 2 --B 5 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/generate_dataset.py         --Nt 64 --Nr 4 --K 10 --dk 2 --B 5 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u learn_UW/train_main.py               --Nt 64 --Nr 4 --K 10 --dk 2 --B 5 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
