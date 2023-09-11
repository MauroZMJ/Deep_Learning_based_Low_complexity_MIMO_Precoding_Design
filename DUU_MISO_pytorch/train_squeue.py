import os
import subprocess as sp
import time
from sys import argv

'''prune experiment'''
#sp.run(['bash', '-c', 'python generate_dataset.py  --Nt 64 --Nr 4 --dk 2 --K 8 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
sp.run(['bash', '-c', 'python train.py             --Nt 64 --Nr 4 --dk 2 --K 8 --SNR 0 --SNR_channel 100 --gpu 1 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
sp.run(['bash', '-c', 'python prune_naive.py       --Nt 64 --Nr 4 --dk 2 --K 8 --SNR 0 --SNR_channel 100 --gpu 1 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])

#sp.run(['bash', '-c', 'python generate_dataset.py  --Nt 64 --Nr 4 --dk 2 --K 10 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
sp.run(['bash', '-c', 'python train.py             --Nt 64 --Nr 4 --dk 2 --K 10 --SNR 0 --SNR_channel 100 --gpu 1 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
sp.run(['bash', '-c', 'python prune_naive.py       --Nt 64 --Nr 4 --dk 2 --K 10 --SNR 0 --SNR_channel 100 --gpu 1 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])

#sp.run(['bash', '-c', 'python generate_dataset.py  --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
sp.run(['bash', '-c', 'python train.py             --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --SNR_channel 100 --gpu 1 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
sp.run(['bash', '-c', 'python prune_naive.py       --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --SNR_channel 100 --gpu 1 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])

#sp.run(['bash', '-c', 'python generate_dataset.py  --Nt 64 --Nr 4 --dk 2 --K 14 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
sp.run(['bash', '-c', 'python train.py             --Nt 64 --Nr 4 --dk 2 --K 14 --SNR 0 --SNR_channel 100 --gpu 1 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
sp.run(['bash', '-c', 'python prune_naive.py       --Nt 64 --Nr 4 --dk 2 --K 14 --SNR 0 --SNR_channel 100 --gpu 1 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])

''''SNR_channel -7dB,SNR 0dB experiment'''''
# sp.run(['bash', '-c', 'python learn_TF.py  --Nt 64 --Nr 4 --dk 1 --K 10 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python learn_TF.py  --Nt 64 --Nr 4 --dk 2 --K 10 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python learn_TF.py  --Nt 64 --Nr 4 --dk 3 --K 10 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python learn_TF.py  --Nt 64 --Nr 4 --dk 4 --K 10 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python learn_TF.py  --Nt 64 --Nr 4 --dk 2 --K 8 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python learn_TF.py  --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python learn_TF.py  --Nt 64 --Nr 4 --dk 2 --K 14 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python learn_TF.py  --Nt 64 --Nr 4 --dk 2 --K 16 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])

#sp.run(['bash', '-c', 'python learn_TF.py  --Nt 64 --Nr 4 --dk 1 --K 10 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
#sp.run(['bash', '-c', 'python learn_TF.py  --Nt 64 --Nr 4 --dk 1 --K 10 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])


"""SNR_channel 3dB,SNR 10dB experiments"""
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 1 --K 10 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 10 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 3 --K 10 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 4 --K 10 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
#
#
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 8 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 14 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 16 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
#
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 1 --K 10 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 10 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 3 --K 10 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 4 --K 10 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
#
#
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 8 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 14 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 16 --SNR 0 --SNR_channel -7 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
#
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 12 --SNR -5 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 5 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 10 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
# sp.run(['bash', '-c', 'python WMMSE_test.py  --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 15 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2'])
#
