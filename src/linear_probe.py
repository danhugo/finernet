import utils

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchmetrics
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import datetime

torch.backends.cudnn.deterministic = True

from datasets import DatasetTransforms
from configs import configs as cfg

from models import *

DatasetTransforms = DatasetTransforms()
LOG_NAME = f'class_{cfg.DATA.DATASET}_{cfg.EXP.MODEL}_{cfg.EXP.CLASS.FEATURE}' + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")

utils.init_logger(name=LOG_NAME)

NUM_EPOCH = cfg.EXP.CLASS.EPOCH

config = {
    'epoch': NUM_EPOCH,
    'model': cfg.EXP.MODEL,
    'batch_size': cfg.EXP.CLASS.BS,
    'feature': cfg.EXP.CLASS.FEATURE,
}

def get_dataset():
    """Return trainset, testset"""
    trainset = torchvision.datasets.ImageFolder(
        root=cfg.DATA.TRAIN_PATH,
        transform=DatasetTransforms.train_data_transforms,
    )

    testset = torchvision.datasets.ImageFolder(
        root=cfg.DATA.TEST_PATH, 
        transform=DatasetTransforms.test_data_transforms,
    )
    
    return trainset, testset

def get_dataloader(trainset, testset):
    """Return trainloader, testloader"""
    train_loader = DataLoader(
        trainset, 
        batch_size=cfg.EXP.CLASS.BS, 
        shuffle=True,
        num_workers=8,
        pin_memory=True)
    
    test_loader = DataLoader(
        testset, 
        batch_size=cfg.EXP.CLASS.BS, 
        shuffle=False,
        num_workers=8,
        pin_memory=True)

    return train_loader, test_loader


def update_partial_weight(model):
    advnet_path = cfg.EXP.ADVNET.PATH
    finernet_dict = torch.load(advnet_path)['model_state_dict']

    initial_path = cfg.EXP.CLASS.PATH
    initial_dict = torch.load(initial_path)

    load_dict = dict()
    finernet_keys = finernet_dict.keys()
    initial_keys = initial_dict.keys()

    for finernet_key, initial_key in zip(finernet_keys, initial_keys):
        if finernet_dict[finernet_key].shape == initial_dict[initial_key].shape:
            load_dict[initial_key] = finernet_dict[finernet_key]


    alpha = cfg.EXP.CLASS.ALPHA #0.2 88.30
    for key in load_dict.keys():
        load_weight = load_dict.get(key)
        if load_weight is not None:
            initial_dict[key] = alpha * initial_dict[key] + (1-alpha) * load_weight

    for key in ['fc.weight', 'fc.bias', 'head.weight', 'head.bias']:
        if initial_dict.get(key) != None:
            initial_dict.pop(key)

    msg = model.load_state_dict(initial_dict, strict=False)
    utils.logger('initial weight: {} - update weight: {}'.format(cfg.EXP.CLASS.PATH, cfg.EXP.ADVNET.PATH))
    utils.logger('portion of initial weight: alpha {} - load msg {}'.format(alpha, msg), level='warning')

    return model

def get_model():
    if cfg.EXP.MODEL.__contains__('resnet'):
        model = Resnet(arch=cfg.EXP.MODEL, pretrained=cfg.EXP.CLASS.FEATURE, num_classes=cfg.DATA.NUM_CLASSES)
        head_name = 'fc'

    if cfg.EXP.MODEL.__contains__('vit'):
        model = ViT(arch=cfg.EXP.MODEL, pretrained=cfg.EXP.CLASS.FEATURE, num_classes=cfg.DATA.NUM_CLASSES)
        head_name = 'head'

    if cfg.EXP.CLASS.FEATURE == 'mix':
        model = update_partial_weight(model)

    if cfg.EXP.CLASS.FREEZE:
        for name, p in model.named_parameters():
            if not name.startswith(head_name):
                p.requires_grad = False
            else:
                p.requires_grad = True

    head_params = getattr(model, head_name).parameters()
    base_params = (param for name, param in model.named_parameters() if not name.startswith(head_name))

    return {
        'model': model,
        'head_params': head_params,
        'base_params': base_params
    }

class Trainer():
    def __init__(
            self,
            gpu_id,
            trainloader,
            testloader,
            model_config):
        self.gpu_id = gpu_id
        self.model = model_config['model'].to(self.gpu_id)
        self.trainloader =  trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()
        self.optimizer = optim.SGD(
            [
                {'params': model_config['head_params'], 'lr': cfg.EXP.CLASS.LR},
                {'params': model_config['base_params'], 'lr': cfg.EXP.CLASS.LR}
            ], 
            momentum=0.9,
            weight_decay=1e-3) #1e-4
        
        # self.optimizer = optim.Adam(
        #     [
        #         {'params': model_config['head_params'], 'lr': cfg.EXP.CLASS.LR},
        #         {'params': model_config['base_params'], 'lr': cfg.EXP.CLASS.LR}
        #     ], 
        #     ) #1e-4
        
        self.scheduler = None
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10, verbose=True)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 50], gamma=0.5)
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=[cfg.EXP.CLASS.LR, cfg.EXP.CLASS.LR], div_factor=10, total_steps=cfg.EXP.CLASS.EPOCH * len(trainloader), base_momentum=0.9)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.985 ** epoch)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=NUM_EPOCH, eta_min=1e-4)
            
        self.train_acc_fn = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.DATA.NUM_CLASSES).to(self.gpu_id)
        self.test_acc_fn = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.DATA.NUM_CLASSES).to(self.gpu_id)

    def _run_batch(self, queries, targets, phase):
        # targets = targets.to(self.gpu_id)
        # outputs = self.model(queries.to(self.gpu_id))
        # loss = self.criterion(outputs, targets)
        # if phase == 'train':
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()
        #     if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
        #         self.scheduler.step()

        #     classifier_lr = self.optimizer.param_groups[0]['lr']
        #     base_params_lr = self.optimizer.param_groups[1]['lr']
        #     wandb.log({'classifier_lr': classifier_lr, 'base_params_lr': base_params_lr})
        #     self.train_acc_fn.update(outputs, targets)
        # else:
        #     self.test_acc_fn.update(outputs, targets)
        # return loss.item()
    
        with autocast():
            targets = targets.to(self.gpu_id)
            outputs = self.model(queries.to(self.gpu_id))
            loss = self.criterion(outputs, targets)
        if phase == 'train':
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            classifier_lr = self.optimizer.param_groups[0]['lr']
            base_params_lr = self.optimizer.param_groups[1]['lr']
            wandb.log({'classifier_lr': classifier_lr, 'base_params_lr': base_params_lr})
            self.train_acc_fn.update(outputs, targets)
        else:
            self.test_acc_fn.update(outputs, targets)
        return loss.item()

    def _run_epoch(self, epoch):
        phase_groups = [
            ('train', self.trainloader, self.train_acc_fn),
            ('test', self.testloader, self.test_acc_fn)
        ]
        for phase, dataloader, metric in phase_groups:
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()

            with torch.set_grad_enabled(phase == 'train'):
                loss = 0.0
                for queries, targets in tqdm(dataloader):
                    batch_loss = self._run_batch(queries, targets, phase)
                    loss += batch_loss
                
                epoch_loss = loss / len(dataloader)
                epoch_acc = metric.compute().item()

                if phase == 'test':
                    self.test_acc = epoch_acc

                utils.logger('phase {}: loss {:.4f} acc {:.4f}'.format(phase, epoch_loss, epoch_acc))
                wandb.log({'{}_accuracy'.format(phase): epoch_acc * 100, '{}_loss'.format(phase): epoch_loss})
                
                metric.reset()

    def _save_epoch(self):
        ckp = self.model.state_dict()
        torch.save(ckp, f"best_models/{wandb.run.name}.pt")

    def train(self, num_epochs):
        self.best_acc = 0.0
        for epoch in range(num_epochs):
            utils.logger('Epoch: {}/{}'.format(epoch+1, num_epochs))
            self._run_epoch(epoch)
            if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.test_acc)
                else:
                    self.scheduler.step()

            if self.best_acc < self.test_acc:
                self.best_acc = self.test_acc
                self._save_epoch()
            utils.logger('best test acc: {:.4f}'.format(self.best_acc))


class TrainerDP(Trainer):
    def __init__(
        self,
        trainloader,
        testloader,
        model_config,
    ):
        self.gpu_id = "cuda"
        super().__init__(self.gpu_id, trainloader, testloader, model_config) 
        
        self.model = nn.DataParallel(self.model)   

def main():
    utils.init_wandb(
        entity=cfg.WANDB.ENTITY,
        project=cfg.WANDB.PROJECT,
        run_name_prefix=f'class-{cfg.DATA.DATASET}-{cfg.EXP.MODEL}-{cfg.EXP.CLASS.FEATURE}',
        config=config,
        mode=cfg.WANDB.MODE,
    )


    wandb.save('advnet/linear_probe.py', policy='now')
    wandb.save('datasets.py', policy='now')
    wandb.save('configs.yaml', policy='now')
    wandb.save(f'logs/{LOG_NAME}.log', policy='live')
    wandb.save('custom_resnet.py', policy='now')

    trainset, testset = get_dataset()
    trainloader, testloader = get_dataloader(trainset, testset)
    model_config = get_model()
    trainer = Trainer(torch.cuda.current_device(), trainloader, testloader, model_config)
    # trainer = TrainerDP(trainloader, testloader, model_config)
    trainer.train(cfg.EXP.CLASS.EPOCH)
    
    wandb.finish()

if __name__ == "__main__":
    main()