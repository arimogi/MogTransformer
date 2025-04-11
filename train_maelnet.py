import argparse
from exp.exp_main_anomaly_SL import Exp_Anomaly_Detection_SL
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Train MaelNet on MSL dataset')

    # Model & Data
    parser.add_argument('--model', type=str, default='MaelNet')
    parser.add_argument('--data', type=str, default='MSL')
    parser.add_argument('--enc_in', type=int, default=55)   # Sesuaikan dengan fitur data MSL
    parser.add_argument('--dec_in', type=int, default=55)
    parser.add_argument('--c_out', type=int, default=55)

    # Window dan Attention
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)

    # NS Transformer Specific
    parser.add_argument('--p_hidden_dims', nargs='+', type=int, default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--output_attention', action='store_true')

    # Training
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--k', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel input size')

    # Device
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--device_ids', type=int, nargs='+', default=[0])

    # Pathing
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--result_dir', type=str, default='./results/')
    parser.add_argument('--root_path', type=str, default='./data/MSL/')  # pastikan folder MSL ada
    parser.add_argument('--chunk_size', type=int, default=100)

    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='MaelNet_MSL')
    parser.add_argument('--is_slow_learner', type=bool, default=True)
    parser.add_argument('--anomaly_ratio', type=float, default=0.85)
    parser.add_argument('--factor', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--des', type=str, default='TA')
    parser.add_argument('--patch_size', type=int, default=5)
    parser.add_argument('--channel', type=int, default=55)
    parser.add_argument('--itr', type=int, default=1)


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.device_ids[0]}')
    else:
        args.device = torch.device('cpu')

    exp = Exp_Anomaly_Detection_SL(args)
    setting = f"{args.model}_{args.data}_win{args.win_size}_run1"
    exp.train(setting)
