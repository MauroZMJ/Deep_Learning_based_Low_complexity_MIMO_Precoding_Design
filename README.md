# Low Complexity Multi-User MIMO Precoding Design
## Descriptions
This repository contains the code for our TWC work "A Deep Learning-Based Framework for Low Complexity
Multi-User MIMO Precoding Design", available at <https://ieeexplore.ieee.org/document/9834153?denied=>

For any repreduce, further research or development, please kindly cite our TWC journal paper: 
<table><tr><td bgcolor=Gainsboro>M. Zhang, J. Gao and C. Zhong, "A Deep Learning-Based Framework for Low Complexity Multiuser MIMO Precoding Design," in IEEE Transactions on Wireless Communications, vol. 21, no. 12, pp. 11193-11206, Dec. 2022, doi: 10.1109/TWC.2022.3190435.</td></tr></table>

## Requirements
    torch==1.8.0
    tensorflow==2.3.0
    python==3.8.0
    hdf5storage
    sklearn
    numpy
## Structures
There are several folders: ''`differ_ds`', '''`DUU_MISO_pytorch`',''`execution_time_test`',''`imperfect_CSI`',''`learn_UW`',''`multi_RB`',''`single_RB`',''`zero_patch_new`'
### Generate  dataset
The folder ''`generate_dataset`'' contains the code for dataset generation. 
- generate_channel.py: generate the MIMO channels
```python
python3 generate_dataset/generate_channel.py 
```
  
- generate_dataset.py: generate the labels for learning
```python
python3 generate_dataset/generate_dataset.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
```

### Learning based precoding for single RB MIMO system
The folder ''`single_RB`'' contains the code for Fig. 2. 
- data_preprocess.py: some functions for dataset preprocessing, including transforming method in Section III-B
- train_main.py: main training functions for the method presented in Fig 2. 
  Some baseline schemes (including ZF, WMMSE) are also provided.   
```python
python3 -u single_RB/train_main.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
```

### Learning based low complexity precoding design for OFDM-MIMO system
The folder ''`multi_RB`'' contains the code for the proposed method in Fig. 4. 
- data_preprocess.py: some functions for dataset preprocessing, including transforming method in Section V
- train_main.py: main training functions for the method presented in Fig 4. 
  Some baseline schemes (including ZF, WMMSE) are also provided.   
```python
python3 -u multi_RB/train_main.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
```
### Varying Number of User Streams
The folder ''`differ_ds`'' contains the code for evaluating the performance when $d_k \neq d_j$. 
```python
python3 -u differ_ds/learn_from_bar_merge_rb.py --Nt 64 --Nr 4 --K 8 --dk 2 --B 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
```
### Varying Number of Users
The folder ''`zero_patch`'' contains the code for evaluating the performance when user number varies. 
```python
python3 -u zero_patch/train.py  --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --B 1 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2 
```

### Imperfect CSI 
The folder ''`imperfect_CSI`'' contains the code for evaluating the performance when user number varies. 
```python
python3 -u zero_patch/train_main.py  --Nt 64 --Nr 4 --K 12 --dk 2 --B 1 --SNR 0 --SNR_channel 10 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1 
```

### Network Pruning
The folder ''`DUU_MISO_pytorch`'' contains the code for evaluating the performance when user number varies. 
```python
python3 -u DUU_MISO_pytorch/prune_naive.py --Nt 64 --Nr 4 --dk 2 --K 16 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu  --batch_size 200 --epoch 1000 --factor 2
```
