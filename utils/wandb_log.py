import wandb
from utils.distributed_gpus import is_main_process

def init_wandb(entity, project, run_name_prefix='experiment', config=None, mode='disabled', **kwargs):
    """
    mode: disabled, online, offline
    """
    if is_main_process():
        api = wandb.Api()
        runs = api.runs(f'{entity}/{project}')
        run_number = len(runs) + 1
        run_name =  f'{run_name_prefix}-{run_number}'

        wandb.init(
            mode=mode,
            project=project,
            entity=entity,
            config=config,
            name=run_name,
            **kwargs
        )

def log_wandb(*args, **kwargs):
    if is_main_process():
        wandb.log(*args, **kwargs)