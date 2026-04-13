import builtins
import datetime
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
from collections import deque

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0005,path_save = "/home/tree/Work_area/AMP_Design/save/best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0         
        self.best_score = None    
        self.early_stop = False  
        self.best_model_state = None 
        self.path_save = path_save
    def __call__(self, metric, model):
        
        score = metric
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self._save_best()
        else:
            self.best_score = score
            self._save_checkpoint(model)
            self.counter = 0 
    def _save_checkpoint(self, model):
        self.best_model_state = model.state_dict() 

    def _save_best(self):
        to_save ={
            'model_state_dict': self.best_model_state,
            'score': self.best_score,
        }
        torch.save(to_save, self.path_save)

class SmoothedValue:
    """Track a series of values and provide access to smoothed statistics.
    Single-GPU version (no distributed synchronization).
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """Add a new value.
        
        Args:
            value (float): The metric value.
            n (int): Number of samples this value represents (for weighting).
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n 
    @property
    def median(self):
        if len(self.deque) == 0:
            return 0.0
        d = sorted(self.deque)
        mid = len(d) // 2
        if len(d) % 2 == 0:
            return (d[mid - 1] + d[mid]) / 2
        else:
            return d[mid]

    @property
    def avg(self):
        if len(self.deque) == 0:
            return 0.0
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self):
        if len(self.deque) == 0:
            return 0.0
        return max(self.deque)

    @property
    def value(self):
        if len(self.deque) == 0:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, (float, int)):
                pass
            elif hasattr(v, 'item'):
                v = v.item()
            else:
                raise TypeError(f"Unsupported type for metric '{k}': {type(v)}")
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time)
                    ))
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def construct_optimizer_V1(model, weight_decay=0, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
    
def construct_optimizer_V2(model, lr, weight_decay=0, lr_multiple=1.0):
    freeze_no_decay = []      
    freeze_decay = []        
    unfreeze_no_decay = []    
    unfreeze_decay = []      
    
    for name, param in model.named_parameters():
        is_frozen = not param.requires_grad
        is_no_decay = (
            len(param.shape) == 1 or         
            name.endswith(".bias") or      
            'norm' in name.lower()   
        )
        if is_frozen:
            if is_no_decay:
                freeze_no_decay.append(param)
            else:
                freeze_decay.append(param)
        else:
            if is_no_decay:
                unfreeze_no_decay.append(param)
            else:
                unfreeze_decay.append(param)
    
    param_groups = []
    
    if freeze_no_decay:
        param_groups.append({
            'params': freeze_no_decay,
            'lr': lr,
            'weight_decay': 0.
        })
    
    if freeze_decay:
        param_groups.append({
            'params': freeze_decay,
            'lr': lr,
            'weight_decay': weight_decay
        })
    
    if unfreeze_no_decay:
        param_groups.append({
            'params': unfreeze_no_decay,
            'lr': lr*lr_multiple,
            'weight_decay': 0.
        })
    
    if unfreeze_decay:
        param_groups.append({
            'params': unfreeze_decay,
            'lr': lr*lr_multiple,
            'weight_decay': weight_decay
        })
    return param_groups

def save_model(args, model, optimizer, epoch,epoch_name=None):
    if epoch_name is None:
        epoch_name = str(epoch)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)

    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    torch.save(to_save, checkpoint_path)

def freeze_layers(model, freeze_patterns=None, unfreeze_patterns=None):
    freeze_patterns = freeze_patterns or []
    unfreeze_patterns = unfreeze_patterns or []
    
    for name, param in model.named_parameters():
        if any(pat in name for pat in unfreeze_patterns):
            param.requires_grad = True
            continue
        if any(pat in name for pat in freeze_patterns):
            param.requires_grad = False