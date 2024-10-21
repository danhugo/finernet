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
from torch.cuda.amp import GradScaler, autocast
# from torchvision.ops import sigmoid_focal_loss

from datasets import BinaryDatasetTransforms, ImageFolderForNNs
from configs import configs as cfg

from models import *
DatasetTransforms = BinaryDatasetTransforms()
torch.backends.cudnn.benchmark = True

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
        drop_last=True,
    )

    return dataloader

def get_dataloader_ddp(dataset):
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.EXP.ADVNET.BS,
        shuffle=False,
        sampler=sampler,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, sampler

def get_model():
    if cfg.EXP.MODEL.__contains__('resnet'):
        model = CNN_finernetwork(model=cfg.EXP.MODEL, pretrained=cfg.EXP.ADVNET.FEATURE)
    
    if cfg.EXP.MODEL.__contains__('vit'):
        model = Transformer_finernetwork(model=cfg.EXP.MODEL, pretrained=cfg.EXP.ADVNET.FEATURE)

    return model
    
def contrastive_loss(label, x, y, p=2):
    margin = 2
    distance = 1 - nn.functional.cosine_similarity(x, y)
    loss = torch.mean((label * torch.pow(distance, p) + 0.5 * (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), p)))
    return loss

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
        self.optimizer = optim.SGD(model.parameters(), lr=cfg.EXP.ADVNET.LR, momentum=0.95, weight_decay=5e-4)
        # self.optimizer = optim.Adam(model.parameters(), lr=cfg.EXP.ADVNET.LR)
        self.scaler = GradScaler()
        
        self.lr_scheduler = None
        # self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer, max_lr=[5e-2],
        #     div_factor=10, total_steps=global_configs.finernet_EPOCH * len(self.trainloader), base_momentum=0.90)

        self.acc_fn = torchmetrics.Accuracy(task='binary').to(self.gpu_id)
            
        self.precision_fn = torchmetrics.Precision(task='binary').to(self.gpu_id)
        self.recall_fn = torchmetrics.Recall(task="binary").to(self.gpu_id)
        self.f1_fn = torchmetrics.F1Score(task="binary").to(self.gpu_id)
        self.pred_yes_sum = torchmetrics.SumMetric().to(self.gpu_id)
        self.target_yes_sum = torchmetrics.SumMetric().to(self.gpu_id)

    def _run_batch(self, queries, explanations, labels):
        self.optimizer.zero_grad()

        with torch.enable_grad():
            # output generated after agg_branch (MLPs in finernet)
            with autocast():
                output, input_feat, exp_feat, sim = self.model(queries, explanations)
                p = torch.sigmoid(output)
                preds = (p >= 0.5).long().squeeze()
                # cstr_loss = contrastive_loss(labels.float(), input_feat, exp_feat, p=1)
                loss = self.criterion(output.squeeze(), labels.float())
                # loss = sigmoid_focal_loss(output.squeeze(), labels.float(), reduction='mean', gamma=0.5, alpha=0.7)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # print(self.model.binary_layer.net[0].weight.grad.mean())
            
            # output, input_feat, exp_feat, _ = self.model(queries, explanations)
            # p = torch.sigmoid(output)
            # preds = (p >= 0.5).long().squeeze()
            # # cstr_loss = contrastive_loss(labels.float(), input_feat, exp_feat, p=1)
            # loss = self.criterion(output.squeeze(), labels.float())
            # # loss = sigmoid_focal_loss(output.squeeze(), labels.float(), reduction='mean', gamma=0.5, alpha=0.7)

            # loss.backward()
            # self.optimizer.step()
            
            if self.lr_scheduler is not None and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.lr_scheduler.step()

            self.acc_fn.update(preds, labels)
            self.precision_fn.update(preds, labels)
            self.recall_fn.update(preds, labels)
            self.f1_fn.update(preds, labels)
            self.pred_yes_sum.update(preds)
            self.target_yes_sum.update(labels)
            return loss.item()

    def _run_epoch(self, epoch: int):
        loss = 0.0

        for queries, explanations, labels, _, _ in tqdm(self.trainloader, disable = not utils.is_main_process()):
            labels = labels.to(self.gpu_id)  # label: (1/0 - yes/no) for query and its NN
            explanations = explanations.to(self.gpu_id)
            queries = queries.to(self.gpu_id)

            loss_batch = self._run_batch(queries, explanations, labels)
            loss += loss_batch

        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.lr_scheduler.step()

        self.epoch_acc = self.acc_fn.compute().item()
        self.epoch_loss = loss / len(self.trainloader)
        self.pred_yes = self.pred_yes_sum.compute().item() / self.n_dataset
        self.epoch_f1 = self.f1_fn.compute().item() 

        utils.logger('epoch: {} - loss {:.4f} - acc {:.4f} - best acc {:.4f} - precision {:.2f} - recall {:.2f} - f1 {:.2f} - pred yes {:.2f} - target yes {:.2f}'.format(
            epoch,
            self.epoch_loss,
            self.epoch_acc,
            self.best_acc,
            self.precision_fn.compute().item(),
            self.recall_fn.compute().item(),
            self.epoch_f1,
            self.pred_yes,
            self.target_yes_sum.compute().item() / self.n_dataset,
        ))

        utils.log_wandb({
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'train_loss': self.epoch_loss,
            'train_accuracy': self.epoch_acc * 100,
            })
        
        self.acc_fn.reset()
        self.precision_fn.reset()
        self.recall_fn.reset()
        self.f1_fn.reset()
        self.pred_yes_sum.reset()
        self.target_yes_sum.reset()

    def _save_checkpoint(self, epoch: int, save=False):
        ckp = self.model.state_dict()
        ckpt_path = 'best_models/{}.pt'.format(wandb.run.name)
        if save==True:
            ckpt_path = 'best_models/{}_{}.pt'.format(wandb.run.name, epoch)

        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': ckp,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': self.epoch_acc,
            'val_yes_ratio':  self.pred_yes,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        if self.lr_scheduler is not None:
            save_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        torch.save(save_dict, ckpt_path)        
    
    def train(self, range_epoch):
        since = time.time()
        self.model.train()
        best_f1 = 0.0
        self.best_acc = 0.0
        for epoch in range_epoch:
            self._run_epoch(epoch)
            if self.epoch_f1 > best_f1:
                best_f1  = self.epoch_f1
                self.best_acc = self.epoch_acc
                self._save_checkpoint(epoch)
            
            # if epoch > 20 and epoch % 10 == 0:
            #     self._save_checkpoint(epoch, save=True)

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
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[gpu_id])
        self.sampler = sampler

    def train(self, range_epoch):
        since = time.time()
        self.model.train()
        best_f1 = 0.0
        self.best_acc = 0.0
        for epoch in range_epoch:
            self.sampler.set_epoch(epoch) # comment out to disable shuffling
            self._run_epoch(epoch)
            if self.epoch_f1 > best_f1:
                best_f1  = self.epoch_f1
                self.best_acc = self.epoch_acc
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

    wandb.save('advnet/model_training.py', policy='now')
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