import torch.distributed as dist
import os

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '55555'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    if not is_distributed():
        return True
    return dist.get_rank() == 0