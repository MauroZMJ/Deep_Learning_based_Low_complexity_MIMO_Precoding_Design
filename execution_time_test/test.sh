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
cd /mnts2d/diis_data1/zmj/LCP_paper/

#python3 -u generate_dataset/generate_dataset.py --Nt 16 --Nr 2 --K 4 --dk 1 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 8 --dk 2 --B 1 --CNN1_prune 0 --CNN2_prune 0 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 8 --dk 2 --B 1 --CNN1_prune 8 --CNN2_prune 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 8 --dk 2 --B 1 --CNN1_prune 12 --CNN2_prune 6 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 8 --dk 2 --B 1 --CNN1_prune 15 --CNN2_prune 7 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1

python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --CNN1_prune 0 --CNN2_prune 0 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --CNN1_prune 8 --CNN2_prune 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --CNN1_prune 12 --CNN2_prune 6 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --CNN1_prune 15 --CNN2_prune 7 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1

python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 12 --dk 2 --B 1 --CNN1_prune 0 --CNN2_prune 0 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 12 --dk 2 --B 1 --CNN1_prune 8 --CNN2_prune 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 12 --dk 2 --B 1 --CNN1_prune 12 --CNN2_prune 6 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 12 --dk 2 --B 1 --CNN1_prune 15 --CNN2_prune 7 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1

python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 14 --dk 2 --B 1 --CNN1_prune 0 --CNN2_prune 0 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 14 --dk 2 --B 1 --CNN1_prune 8 --CNN2_prune 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 14 --dk 2 --B 1 --CNN1_prune 12 --CNN2_prune 6 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
python3 -u ./execution_time_test/test_run_time_prune.py --Nt 64 --Nr 4 --K 14 --dk 2 --B 1 --CNN1_prune 15 --CNN2_prune 7 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1

# python3 -u ./execution_time_test/train_main.py --Nt 16 --Nr 2 --K 5 --dk 1 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1

# python3 -u ./execution_time_test/train_main.py --Nt 16 --Nr 2 --K 6 --dk 1 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1

# python3 -u ./execution_time_test/train_main.py --Nt 16 --Nr 2 --K 7 --dk 1 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1

# python3 -u ./execution_time_test/train_main.py --Nt 16 --Nr 2 --K 8 --dk 1 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1

# python3 -u ./execution_time_test/train_main.py --Nt 64 --Nr 4 --K 8 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1

# python3 -u ./execution_time_test/train_main.py --Nt 64 --Nr 4 --K 9 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1

# python3 -u ./execution_time_test/train_main.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1

# python3 -u ./execution_time_test/train_main.py --Nt 64 --Nr 4 --K 11 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1

# python3 -u ./execution_time_test/train_main.py --Nt 64 --Nr 4 --K 12 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
# python3 -u ./execution_time_test/train_main.py --Nt 64 --Nr 4 --K 13 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
# python3 -u ./execution_time_test/train_main.py --Nt 64 --Nr 4 --K 14 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
# python3 -u ./execution_time_test/train_main.py --Nt 64 --Nr 4 --K 15 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
# python3 -u ./execution_time_test/train_main.py --Nt 64 --Nr 4 --K 16 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1
