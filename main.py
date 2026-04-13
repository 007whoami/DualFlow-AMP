import argparse
import datetime
import os
import ast
import time
import random
from typing import Tuple, Optional, List, Union
from pathlib import Path
import json
import pandas as pd
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch._inductor import config

import util.misc as misc

from engine import train_one_epoch,evalution
from denoiser import Denoiser
from util.normalize_phychem import AMPFeatureNormalizer

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


class AMPDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        amp_embed_dir: str,
        reference_lib_json: str,
        phychem_csv: str,
        family_dir: str,
    ):
        super().__init__()
        self.phychem_data = pd.read_csv(phychem_csv)
        self.n_samples = len(self.phychem_data)
        self.normalizer = AMPFeatureNormalizer()
        self.normalizer.fit(self.phychem_data)
        self.normalizer.save(  "/home/tree/Work_area/AMP_Design_Github/output_dir/phychem_normalizer.pkl")
        self.phychem_data = self.normalizer.transform(self.phychem_data)

        self.amp_embed_dir = Path(amp_embed_dir)
        self.family_dir = Path(family_dir)
        
        with open(reference_lib_json, 'r') as f:
            self.reference_library = json.load(f)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor,np.ndarray, torch.Tensor]:
        amp_embed = torch.load(self.amp_embed_dir / f'{index}.pt')
        phychem = self.phychem_data[index]
        refs = self.reference_library[str(index)]
        if len(refs) >= 1:
            ref_embeds_id = random.choice(refs)
            ref_embeds = torch.load(self.family_dir / f'{ref_embeds_id}.pt')
        else:
            ref_embeds = amp_embed
        return amp_embed, phychem, ref_embeds #type:ignore

def main(args):
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(',', ',\n'))
    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    config.max_autotune_gemm = False
    torch.set_float32_matmul_precision('high')

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None


    data_path = Path(args.data_path)
    amp_emb_dir = data_path/"embeddings_amp"
    reference_lib_json = data_path/"output_mapping.json"
    phychem_csv = data_path/"amp_phychem_df.csv"
    family_dir = data_path/"embeddings_family"
    dataset_train = AMPDataset(
                               f"{amp_emb_dir}",
                               f"{reference_lib_json}",
                               f"{phychem_csv}",
                               f"{family_dir}",
                               )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=True,
    )
    
    # Create denoiser
    model = Denoiser(args)
    if args.pretrain :
        freeze_layer = ["y_embedder"]
        misc.freeze_layers(model, freeze_patterns=freeze_layer)
        param_groups = misc.construct_optimizer_V1(model, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
        print(optimizer)
        print("Going Pretrain")
    else:
        print("Going Fine-tnuing")
        checkpoint_path = os.path.join(args.resume, "/home/tree/Work_area/AMP_Design_Github/output_dir/checkpoint-last.pth") if args.resume else None
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['model'])
            del checkpoint

            freeze_layer = ["f_embedder","cross_attn","mlp_embedding"]    
            misc.freeze_layers(model, freeze_patterns=freeze_layer)

            param_groups = misc.construct_optimizer_V2(model, args.lr,args.weight_decay,lr_multiple=0.1)
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
            
            misc.freeze_layers(model, unfreeze_patterns=freeze_layer)
            
            print(optimizer)
            print("Going Fine-tnuing")
        else:
            print("False , you are fine-tuning but dont have pretrained model")
            exit()
    
    model.to(device)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    # early_stop_manger = misc.EarlyStopping(patience=7,min_delta=0.001)
    early_stop_manger = None
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch( model, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args,early_stop_manger=early_stop_manger)

        # Save checkpoint periodically or finish  training
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            print("Saving checkpoint")
            misc.save_model(
                args=args,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )
        if early_stop_manger is not None:
            if early_stop_manger.early_stop:
                print("Early stopping")
                break
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)
    if log_writer is not None:
        log_writer.close() 

if __name__ == '__main__':
    
    # Parse arguments and load global stats
    args = get_args_parser().parse_args()
    states = torch.load( str(Path(args.data_path) /"/home/tree/Work_area/AMP_Design_Github/Data/all_embedding_statistic.pth" ))
    args.global_stats = states
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
