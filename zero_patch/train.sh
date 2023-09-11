#python3 -u generate_dataset.py --Nt 64 --Nr 4 --K 12 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 -u train.py            --Nt 64 --Nr 4 --dk 2 --K 12 --SNR 0 --B 1 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2
