import utils

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchmetrics
import time
from tqdm import tqdm
import wandb
import datetime

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
# from torchvision.ops import sigmoid_focal_loss

from datasets import BinaryDatasetTransforms, ImageFolderForNNs
from configs import configs as cfg

from models import *
DatasetTransforms = BinaryDatasetTransforms()
WANDB_SESS_NAME = f"finernet-{cfg.DATA.DATASET}"

RUN_NAME='{}-{}-{}'.format(
                WANDB_SESS_NAME,
                cfg.EXP.ADVNET.FEATURE,
                cfg.EXP.MODEL,
            )

LOG_NAME = RUN_NAME + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")

def get_dataset():
    dataset = ImageFolderForNNs(
        cfg.DATA.AUG_TRAIN_DATA_PATH, 
        train=True, 
        transform=DatasetTransforms.train_data_transforms
    )

    return dataset

def get_dataloader_single(dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.EXP.ADVNET.BS,
        shuffle=True,  
        num_workers=16,
        pin_memory=True,
        drop_last=True # avoid the asymmetric distribution of the last batch
    )

    return dataloader

def get_dataloader_ddp(dataset):
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.EXP.ADVNET.BS,
        shuffle=False,
        sampler=sampler,
        num_workers=8,
        drop_last=True
    )

    return dataloader, sampler

def get_model():
    model = ADVNET(model=cfg.EXP.MODEL, pretrained=cfg.EXP.ADVNET.FEATURE)
    return model
    
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive, torch.mean(pos), torch.mean(neg)

class TrainerSingle():
    def __init__(
        self,
        gpu_id: int,
        model: nn.Module,
        trainloader: DataLoader,
    ):
        self.gpu_id = gpu_id

        self.model = model.to(self.gpu_id)
        self.trainloader = trainloader
        self.n_dataset = len(self.trainloader.dataset)
        # pos_weight = torch.tensor([15/7])
        self.criterion = nn.BCEWithLogitsLoss().to(self.gpu_id )
        self.contrastive_loss = ContrastiveLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=4e-4, momentum=0.95, weight_decay=5e-4)
        
        self.lr_scheduler = None
        # self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer, max_lr=[5e-2],
        #     div_factor=10, total_steps=global_configs.finernet_EPOCH * len(self.trainloader), base_momentum=0.90)


    def _run_batch(self, queries, explanations, labels):
        self.optimizer.zero_grad()

        with torch.enable_grad():
            # output generated after agg_branch (MLPs in finernet)
            input_feat, exp_feat = self.model(queries, explanations)
           
            norm_input_feat = torch.nn.functional.normalize(input_feat, p=2, dim=1)
            norm_exp_feat = torch.nn.functional.normalize(exp_feat, p=2, dim=1)
            cstr_loss, pos, neg = self.contrastive_loss(norm_input_feat, norm_exp_feat, labels.float())
            
            loss = cstr_loss
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.lr_scheduler.step()
            
            return loss.item(), cstr_loss.item(), pos.item(), neg.item()

    def _run_epoch(self, epoch: int):
        loss = 0.0
        cstr_loss = 0.0
        pos = 0.0
        neg = 0.0

        for queries, explanations, labels, _, _ in tqdm(self.trainloader, disable = not utils.is_main_process()):
            labels = labels.to(self.gpu_id)  # label: (1/0 - yes/no) for query and its NN
            explanations = explanations.to(self.gpu_id)
            queries = queries.to(self.gpu_id)

            loss_batch, cstr_loss_batch, pos_b, neg_b = self._run_batch(queries, explanations, labels)
            loss += loss_batch
            cstr_loss += cstr_loss_batch
            pos += pos_b
            neg += neg_b

        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.lr_scheduler.step()

        self.epoch_loss = loss / len(self.trainloader)

        utils.logger('epoch: {} - loss {:.4f}'.format(
            epoch,
            self.epoch_loss,
        ))

        utils.logger('epoch: {} - contrastive loss {:.4f} - pos {:.4f} - neg {:4f}'.format(epoch, cstr_loss / len(self.trainloader), pos / len(self.trainloader), neg / len(self.trainloader)))

        utils.log_wandb({
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'train_loss': self.epoch_loss,
            'train_accuracy': 0 * 100,
            })
        

    def _save_checkpoint(self, epoch: int):
        ckp = self.model.state_dict()
        ckpt_path = 'best_models/{}.pt'.format(wandb.run.name)

        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': ckp,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': 0,
            'val_yes_ratio':  0,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        if self.lr_scheduler is not None:
            save_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        torch.save(save_dict, ckpt_path)
    
    def train(self, range_epoch):
        since = time.time()
        self.model.train()
        for epoch in range_epoch:
            self._run_epoch(epoch)
            self._save_checkpoint(epoch)

        utils.logger(f'Training complete in {(time.time() - since) // 60:.0f}m {(time.time() - since) % 60:.0f}s')

class TrainerDP(TrainerSingle):
    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
    ):
        self.gpu_id = "cuda"
        super().__init__(self.gpu_id, model, trainloader) 
        
        self.model = nn.DataParallel(self.model)   

class TrainerDDP(TrainerSingle):
    def __init__(
        self, 
        gpu_id: int, 
        model: nn.Module, 
        trainloader: DataLoader, 
        sampler: DistributedSampler
        ):

        super().__init__(gpu_id, model, trainloader)
        torch.cuda.set_device(gpu_id)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[gpu_id])
        self.sampler = sampler

    def train(self, range_epoch):
        since = time.time()
        self.model.train()
        for epoch in range_epoch:
            self.sampler.set_epoch(epoch) # comment out to disable shuffling
            self._run_epoch(epoch)
            if utils.is_main_process():
                self._save_checkpoint(epoch)

        utils.logger(f'Training complete in {(time.time() - since) // 60:.0f}m {(time.time() - since) % 60:.0f}s')

def main_single(gpu_id: int):
    log_name = utils.init_logger(name=LOG_NAME)
    
    config = {
        'train_path': cfg.DATA.AUG_TRAIN_DATA_PATH,
        "num_epochs": cfg.EXP.ADVNET.EPOCH,
        "batch_size": cfg.EXP.ADVNET.BS,
    }

    if WANDB_SESS_NAME is not None:
        utils.init_wandb(
            entity=cfg.WANDB.ENTITY,
            project=cfg.WANDB.PROJECT,
            run_name_prefix='{}-{}-{}'.format(
                WANDB_SESS_NAME,
                cfg.EXP.ADVNET.FEATURE,
                cfg.EXP.MODEL,
            ),
            config=config,
            mode=cfg.WANDB.MODE,
        )
    else:
        utils.init_wandb(
            entity=cfg.WANDB.ENTITY,
            project=cfg.WANDB.PROJECT,
            config=config,
            mode=cfg.WANDB.MODE,
        )

    wandb.save('advnet/model_training_observe_loss.py', policy='now')
    wandb.save('datasets.py', policy='now')
    wandb.save('configs.yaml', policy='now')
    wandb.save('models.py', policy='now')
    wandb.save(f'logs/{log_name}.log', policy='live')

    dataset = get_dataset()
    dataloader = get_dataloader_single(dataset)
    model = get_model()
    trainer = TrainerSingle(gpu_id=gpu_id, model=model, trainloader=dataloader)
    trainer.train(range(cfg.EXP.ADVNET.EPOCH))
    
    wandb.finish()

def main_ddp(rank, world_size):
    utils.setup_ddp(rank, world_size)

    if utils.is_main_process():
        log_name = utils.init_logger(name=LOG_NAME)
        
        config = {
            'train_path': cfg.DATA.AUG_TRAIN_DATA_PATH,
            "num_epochs": cfg.EXP.ADVNET.EPOCH,
            "batch_size": cfg.EXP.ADVNET.BS,
        }

        if WANDB_SESS_NAME is not None:
            utils.init_wandb(
                entity=cfg.WANDB.ENTITY,
                project=cfg.WANDB.PROJECT,
                run_name_prefix='{}-{}-{}'.format(
                    WANDB_SESS_NAME,
                    cfg.EXP.ADVNET.FEATURE,
                    cfg.EXP.MODEL,
                ),
                config=config,
                mode=cfg.WANDB.MODE,
            )
        else:
            utils.init_wandb(
                entity=cfg.WANDB.ENTITY,
                project=cfg.WANDB.PROJECT,
                config=config,
                mode=cfg.WANDB.MODE,
            )

        wandb.save('advnet/model_training.py', policy='now')
        wandb.save('datasets.py', policy='now')
        wandb.save('global_configs.py', policy='now')
        wandb.save('transformer.py', policy='now')
        wandb.save(f'logs/{log_name}.log', policy='live')

    
    dataset = get_dataset()
    dataloader, sampler = get_dataloader_ddp(dataset)
    model = get_model()
    trainer = TrainerDDP(
        gpu_id=rank,
        model=model,
        trainloader=dataloader,
        sampler=sampler,
    )
    trainer.train(range(cfg.EXP.ADVNET.EPOCH))

    if utils.is_main_process():
        wandb.finish()

    utils.cleanup_ddp()

if __name__ == "__main__":
    # world_size = torch.cuda.device_count()
    # mp.spawn(main_ddp, args=(world_size, ), nprocs=world_size)
    main_single(gpu_id=0)