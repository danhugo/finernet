import torch

def save_model(path, model, optimizer, scheduler=None, epoch=None, params_dict=None):
    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }

    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    
    if params_dict is not None:
        for key, value in params_dict.items():
            save_dict[key] = value
    
    torch.save(save_dict, path)

def load_model(path, model, optimizer, scheduler=None):
    save_dict = torch.load(path)
    model.load_state_dict(save_dict['model'])
    optimizer.load_state_dict(save_dict['optimizer'])
    if scheduler is not None and 'scheduler' in save_dict:
        scheduler.load_state_dict(save_dict['scheduler'])
    epoch = save_dict['epoch']
    params_dict = {}
    for key, value in save_dict.items():
        if key not in ['model', 'optimizer', 'scheduler', 'epoch']:
            params_dict[key] = value
    
    if scheduler is not None:
        return model, optimizer, scheduler, epoch, params_dict
    else:
        return model, optimizer, epoch, params_dict