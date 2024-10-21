import utils

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, SumMetric

from models import  Resnet
from datasets import BinaryDatasetTransforms, ImageFolderForNNs
import global_configs
import model as mod

DatasetTransforms = BinaryDatasetTransforms()


def get_dataset():
    dataset = ImageFolderForNNs(
        global_configs.TEST_DATA_PATH, 
        train=False, 
        transform=DatasetTransforms.test_data_transforms)

    return dataset

def get_dataloader(dataset):
    dataloader = DataLoader(
        dataset,
        batch_size= 17,
        shuffle=False,  # turn shuffle to False
        num_workers=16,
        pin_memory=True,
        drop_last=False  # Do not remove drop last because it affects performance
    )
    return dataloader

def _load_finernet_rn50(model, load_path):
    save_dict = torch.load(load_path)
    finernet_dict = save_dict['model_state_dict']
    utils.logger('model accuracy {} epoch {}'.format(
        save_dict['val_acc'],
        save_dict['epoch']))
    
    load_dict = dict()
    finernet_keys = finernet_dict.keys()
    model_keys = model.state_dict().keys()

    for finernet_key, model_key in zip(finernet_keys, model_keys):
        if finernet_dict[finernet_key].shape == model.state_dict()[model_key].shape:
            load_dict[model_key] = finernet_dict[finernet_key]

    return load_dict

def get_model():
    finernet = mod.CNN_finernetwork(pretrained=None)
    # finernet = mod.CNN_finernetworkHead()

    checkpoint = _load_finernet_rn50(finernet, global_configs.finernet_INFER_PATH)
    msg = finernet.load_state_dict(checkpoint, strict=False)
    utils.logger(f'Load features from {global_configs.finernet_INFER_PATH} to finernet with msg {msg}', level='warning')

    return finernet

class Tester():
    def __init__(
        self,
        dataloader,
        model,
    ):

        self.testloader = dataloader
        self.model = model.cuda()
        self.n_dataset = len(dataloader.dataset)

        self.acc_fn = Accuracy(task='binary').cuda()
        self.precision_fn = Precision(task='binary').cuda()
        self.recall_fn = Recall(task='binary').cuda()
        self.f1_fn = F1Score(task='binary').cuda()
        self.confusion_matrix = ConfusionMatrix(task='binary', num_classes=2).cuda()
        self.pred_yes_sum = SumMetric().cuda()
        self.target_yes_sum = SumMetric().cuda()

    def _run_batch(self, x, explanations, labels):
        x = x.cuda()
        labels = labels.cuda()
        explanations = explanations.cuda()

        output, _, _, _ = self.model(x, explanations)
        model_score = torch.sigmoid(output)
        preds = (model_score >= 0.5).long().squeeze()
        self.acc_fn.update(preds, labels)
        self.precision_fn.update(preds, labels)
        self.recall_fn.update(preds, labels)
        self.f1_fn.update(preds, labels)
        self.confusion_matrix.update(preds, labels)
        self.pred_yes_sum.update(preds)
        self.target_yes_sum.update(labels)

    def _reset_metric(self):
        self.acc_fn.reset()
        self.precision_fn.reset()
        self.recall_fn.reset()
        self.f1_fn.reset()
        self.pred_yes_sum.reset()
        self.target_yes_sum.reset()
        self.confusion_matrix.reset()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for x, explanations, labels, _, _ in tqdm(self.testloader): 
                self._run_batch(x, explanations, labels)
            
            utils.logger('Acc {:.4f} - precision {:.4f} - recall {:.4f} - f1 {:.4f} - pred yes: {:.4f} - target yes: {:.4f}'.format(
                self.acc_fn.compute().item(),
                self.precision_fn.compute().item(),
                self.recall_fn.compute().item(),
                self.f1_fn.compute().item(),
                self.pred_yes_sum.compute().item() / self.n_dataset,
                self.target_yes_sum.compute().item() / self.n_dataset,
            ))
            confusion_matrix = self.confusion_matrix.compute().reshape((1,-1)).squeeze()
            utils.logger('TN {}, FP {}, FN {}, TP {}'.format(*list(confusion_matrix[i].item() for i in range(len(confusion_matrix)))))
        self._reset_metric()

def main():
    dataset = get_dataset()
    testloader = get_dataloader(dataset)
    finernet = get_model()
    tester = Tester(dataloader=testloader, model=finernet)
    tester.test()

if __name__ == '__main__':
    main()