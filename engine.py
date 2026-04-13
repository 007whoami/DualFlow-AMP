import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler  import GradScaler

import util.misc as misc
import util.lr_sched as lr_sched

from util.normalize_phychem import AMPFeatureNormalizer
from back_to_acid.back import EnhancedBackAcidModel

FEATURE_COLUMNS = [
            "Sequence_Length",
            "Net_Charge",      
            "Charge_Density",      
            "GRAVY",   
            "Hydrophobic_Moment",   
            "Aromatic_Ratio",      
            "Proline_Ratio",        
            "Glycine_Ratio",
]
def train_one_epoch(model, data_loader, optimizer, device, epoch,log_writer=None, args=None,early_stop_manger=None):
    """
    train one epoch
    """
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()
    scaler = GradScaler()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels,family_embeding) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if args.pretrain: #type:ignore
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        x = x.to(device, non_blocking=True).to(torch.float32)
        family_embeding = family_embeding.to(device, non_blocking=True).to(torch.float32)
        labels = labels.to(device, non_blocking=True).to(torch.float32)
        
        global_mean = args.global_stats["mean"] #type:ignore
        global_std = args.global_stats["std"] #type:ignore
        global_mean = global_mean.to("cuda")
        global_std = global_std.to("cuda")
        
        x = (x - global_mean)/global_std
        family_embeding = (family_embeding - global_mean)/global_std

        with autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels,family_embeding)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if log_writer is not None and data_iter_step % args.log_freq == 0: #type: ignore
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if early_stop_manger is not None and log_writer is not None:
        early_stop_manger(stats["loss"], model) 

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evalution(model_gen,args,device):
    model_gen.eval()
    model = EnhancedBackAcidModel()
    model = model.to(device)
    path = Path(args.output_dir) / "checkpoint_bta.pth"
    model.load_state_dict(torch.load(path)['model_state_dict'])
    print("Generating ......")
    with torch.no_grad():

        if Path("/home/tree/Work_area/AMP_Design_Github/output_dir/phychem_normalizer.pkl").exists():
            phychem_normalizer = AMPFeatureNormalizer.load("/home/tree/Work_area/AMP_Design_Github/output_dir/phychem_normalizer.pkl")
        else:
            print("Error, There is no phychem_normalizer.pkl file ")
            exit()

        # the order as defined in the normalizer
        specific_condition = [
            19.0, 
            3.5,    
            0.184,  
            -0.2,   
            0.375,   
            0.053, 
            0.105,
            0.105    
        ]
        condition = pd.DataFrame([specific_condition],columns=FEATURE_COLUMNS)
        condition = torch.Tensor(phychem_normalizer.transform(condition))
        condition = condition.expand(args.gen_num, -1)
        condition = condition.to("cuda")

        global_mean = args.global_stats["mean"]
        global_std = args.global_stats["std"]
        global_mean = global_mean.to("cuda")
        global_std = global_std.to("cuda")
        # family = torch.load("/home/tree/Work_area/AMP_Design_Github/single/embedding.pt") # the embeddings of you want to generate,use ESM-2 to extract the embeddings the shape should be [50, 1280]
        # family = family.to("cuda")
        # family = (family - global_mean)/global_std
        # family = family.unsqueeze(0).expand(args.gen_num, -1,-1)

        null_family = model_gen.net.f_embedder.null_token.unsqueeze(0).expand(args.gen_num, -1,-1)
        null_condition = model_gen.net.y_embedder.null_token.unsqueeze(0).expand(args.gen_num, -1)

        # change the parameters in model_gen.generate() to what you want
        x = model_gen.generate(null_condition,null_family)
        x = x*global_std + global_mean
        x = x.to(device)

        x = model(x)
        x = model.back_to_acid(x)
        result = ["".join(sublist) for sublist in x]
        print(result)