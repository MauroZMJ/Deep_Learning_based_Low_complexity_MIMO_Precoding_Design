import argparse
def parge_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--Nt', type=int, default=64, required=True, help='Number of transmitting antennas')
    parser.add_argument('--Nr', type=int, default=4, required=True, help='Number of transmitting antennas')
    parser.add_argument('--dk', type=int, default=2, required=True, help='Number of transmitting antennas')
    parser.add_argument('--K', type=int, default=10, required=True, help='Number of transmitting antennas')
    parser.add_argument('--B', type=int, default=4, required=True, help='Number of transmitting antennas')
    parser.add_argument('--SNR', type=int, default=0, required=True, help='Number of transmitting antennas')
    parser.add_argument('--SNR_channel', type=int, default=100, required=True, help='Number of transmitting antennas')
    parser.add_argument('--gpu', type=int, default=0, required=True, help='Number of transmitting antennas')
    parser.add_argument('--mode', type=str, default='debug', required=True, help='Number of transmitting antennas')
    parser.add_argument('--batch_size', type=int, default=125, required=True, help='Number of transmitting antennas')
    parser.add_argument('--epoch', type=int, default=1000, required=True, help='Number of transmitting antennas')
    parser.add_argument('--factor', type=float, default=1, required=True, help='Number of transmitting antennas')
    parser.add_argument('--test_length', type=int, default=2000, required=False, help='Number of transmitting antennas')
    parser.add_argument('--CNN1_prune',type=int, default=0, required=False, help='Number of transmitting antennas')
    parser.add_argument('--CNN2_prune',type=int, default=0, required=False, help='Number of transmitting antennas')
    parser.add_argument('--CNN3_prune',type=int, default=0, required=False, help='Number of transmitting antennas')
    args = parser.parse_args()


    return args