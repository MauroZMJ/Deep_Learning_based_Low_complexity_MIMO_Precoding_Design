#!/bin/bash
#SBATCH -J AD
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1

source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
module unload cudnn8.0-cuda11.1/
module load cudnn7.6-cuda10.1/7.6.5.32
conda activate pt1.8
cd /mnts2d/diis_data1/zmj/LCP_paper/

# python3 -u DUU_MISO_pytorch/train.py       --Nt 64 --Nr 4 --dk 2 --K 8 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu  --batch_size 200 --epoch 1000 --factor 2
# python3 -u DUU_MISO_pytorch/prune_naive.py --Nt 64 --Nr 4 --dk 2 --K 8 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu  --batch_size 200 --epoch 1000 --factor 2

# python3 -u DUU_MISO_pytorch/train.py       --Nt 64 --Nr 4 --dk 2 --K 10 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu  --batch_size 200 --epoch 1000 --factor 2
# python3 -u DUU_MISO_pytorch/prune_naive.py --Nt 64 --Nr 4 --dk 2 --K 10 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu  --batch_size 200 --epoch 1000 --factor 2

#python3 -u DUU_MISO_pytorch/train.py       --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu  --batch_size 200 --epoch 1000 --factor 2
#python3 -u DUU_MISO_pytorch/prune_naive.py --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu  --batch_size 200 --epoch 1000 --factor 2

#python3 -u DUU_MISO_pytorch/train.py       --Nt 64 --Nr 4 --dk 2 --K 14 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu  --batch_size 200 --epoch 1000 --factor 2
#python3 -u DUU_MISO_pytorch/prune_naive.py --Nt 64 --Nr 4 --dk 2 --K 14 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu  --batch_size 200 --epoch 1000 --factor 2

#python3 -u DUU_MISO_pytorch/train.py       --Nt 64 --Nr 4 --dk 2 --K 16 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu  --batch_size 200 --epoch 1000 --factor 2
python3 -u DUU_MISO_pytorch/prune_naive.py --Nt 64 --Nr 4 --dk 2 --K 16 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu  --batch_size 200 --epoch 1000 --factor 2
