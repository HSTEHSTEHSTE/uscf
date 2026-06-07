import torch

def get_scheduler(type, optimizer, last_epoch, scheduler_args):
    if type == '0':
        return torch.optim.lr_scheduler.ConstantLR(optimizer, 
            factor=scheduler_args['factor'], 
            total_iters=scheduler_args['total_iters'],
            last_epoch=last_epoch
        )