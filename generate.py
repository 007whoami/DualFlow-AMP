import torch
import os
import argparse
from engine import evalution
from denoiser import Denoiser
from pathlib import Path


def get_args_parser():
    parser =  argparse.ArgumentParser('DualFlow-AMP', add_help=False)

    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--model', default='DAMP_B_768', type=str,metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--seq_len', type=int,default=50)
    parser.add_argument('--embeddings_dim',type=int,default=1280,help="the embedding of amp")
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')
    parser.add_argument('--pretrain', action='store_true',help="to decide is pretrain?")

    parser.add_argument('--epochs', default=200, type=int) 
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', 
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=8, type=int,)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument("--gen_num",type=int,default=100,help="Number of generated sequences")
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--cfg_phychem', default=1.0, type=float,)
    parser.add_argument('--cfg_family',default=1.0, type=float,)
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')

    # dataset
    parser.add_argument('--data_path', required=True, type=str,
                        help='Path to the dataset')
    parser.add_argument('--num_features', default=8, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    return parser


args = get_args_parser().parse_args()
states = torch.load( str(Path(args.data_path) /"/home/tree/Work_area/AMP_Design_Github/Data/all_embedding_statistic.pth" ))
args.global_stats = states
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

model = Denoiser(args)
model.to("cuda")
checkpoint_path = "/home/tree/Work_area/AMP_Design_Github/output_dir/checkpoint-last.pth"
if checkpoint_path and os.path.exists(checkpoint_path):
    print("Evaluation")
    checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    evalution(model,args,"cpu")
    exit()