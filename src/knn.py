import utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models import *
from datasets import DatasetTransforms
from helpers import HelperFunctions
from configs import configs as cfg

torch.backends.cudnn.deterministic = True

HelperFunctions = HelperFunctions()
DatasetTransforms = DatasetTransforms()

def get_model():
    model = Resnet(arch=cfg.EXP.MODEL, pretrained=cfg.EXP.CLASS.FEATURE, num_classes=cfg.DATA.NUM_CLASSES)
    # my_model_state_dict = torch.load(
    #     'best_models/class_cub_inat.pt') #85.67

    # msg = resnet.load_state_dict(my_model_state_dict, strict=False)
    return model

def get_model_head():
    model = Resnet(resnet=cfg.EXP.MODEL, pretrained=None, num_classes=cfg.DATA.NUM_CLASSES)
    dim_head = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(dim_head, dim_head),
        torch.nn.ReLU(),
        model.fc
    )
    ckp_path = cfg.EXP.ADVNET.PATH
    load_dict = torch.load(ckp_path)
    fc_keys = [key for key in load_dict.keys() if key.startswith('fc')]
    for key in fc_keys:
        load_dict.pop(key)
            
    msg = model.load_state_dict(load_dict, strict=False)
    utils.logger('{} {}'.format(ckp_path, msg))

    return model

def get_model_head_advnet():
    model = CNN_finernetwork()
    ckp_path = cfg.EXP.ADVNET.PATH
    save_dict = torch.load(ckp_path)
    msg = model.load_state_dict(save_dict['model_state_dict'])
    utils.logger('{} {}'.format(ckp_path, msg))
    return model

def get_feature_extractor(model):
    feature_extractor = nn.Sequential(*list(model.children())[:-1])  # avgpool feature
    return feature_extractor

def get_feature_extractor_head(model):
    class FeatureExtractorHead(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model
            self.conv_layers = nn.Sequential(*list(self.model.children())[:-1])
            self.head = nn.Sequential(*list(self.model.fc.children())[:-1])

        def forward(self, img):
            feat = self.conv_layers(img)
            feat = self.head(feat.squeeze())
            return feat
    
    feature_extractor = FeatureExtractorHead(model)
    return feature_extractor

def get_feature_extractor_head_advnet(model):
    class FeatureExtractorHead(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model
            self.conv_layers = model.conv_layers
            self.head = model.head

        def forward(self, img):
            feat = self.conv_layers(img)
            feat = self.head(feat.squeeze())
            return feat
    
    feature_extractor = FeatureExtractorHead(model)
    return feature_extractor

def get_dataset():
    train_data = ImageFolder(
        root=cfg.DATA.TRAIN_PATH, transform=DatasetTransforms.train_data_transforms
    )

    test_data = ImageFolder(
    root=cfg.DATA.TEST_PATH, transform=DatasetTransforms.test_data_transforms
    )

    return train_data, test_data

def get_dataloader(train_data, test_data):
    train_loader = DataLoader(
        train_data,
        batch_size=512,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        )
    
    test_loader = DataLoader(
        test_data,
        batch_size=512,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    return train_loader, test_loader

class KNNRunner:
    def __init__(self, model, feature_extractor, train_loader, test_loader) -> None:
        self.model = model.cuda()
        self.feature_extractor = feature_extractor.cuda()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_test = len(test_loader.dataset)
    
    def run(self):
        self.feature_extractor.eval()
        all_test_embds = []
        all_test_labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.test_loader)):
                data = data.cuda()

                embeddings = HelperFunctions.to_np(self.feature_extractor(data))
                labels = HelperFunctions.to_np(target)

                all_test_embds.append(embeddings)
                all_test_labels.append(labels)

        all_test_concatted = HelperFunctions.concat(all_test_embds)
        all_test_labels_concatted = HelperFunctions.concat(all_test_labels)

        all_test_concatted = all_test_concatted.reshape(-1, 2048)

        Query = torch.from_numpy(all_test_concatted)
        Query = Query.cuda()
        Query = F.normalize(Query, dim=1)

        similarity_results = []
        target_labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
                data = data.cuda()
                labels = HelperFunctions.to_np(target)

                embeddings = self.feature_extractor(data)
                embeddings = embeddings.view(-1, 2048)
                # cosine similarity
                embeddings = F.normalize(embeddings, dim=1)
                q_results = torch.einsum("id,jd->ij", Query, embeddings).to("cpu") # dot product

                similarity_results.append(q_results)
                target_labels.append(target)

        # Convert to numpy arrays
        train_labels_np = torch.cat(target_labels, -1)
        test_labels_np = np.concatenate(all_test_labels)

        similarity_results = torch.cat(similarity_results, 1)

        # Compute the top-1 accuracy of KNNs, save the KNN dictionary
        scores = {}
        K_test = [5, 10, 20, 50, 100, 200]

        for K in K_test:
            correct_cnt = 0
            for i in tqdm(range(self.num_test)):
                concat_ts = similarity_results[i].cuda()
                # get top k similarity scores
                sorted_ts = torch.argsort(concat_ts).cuda()
                sorted_topk = sorted_ts[-K:]
                scores[i] = torch.flip(
                    sorted_topk, dims=[0]
                )  # Move the closest to the head

                gt_id = test_labels_np[i]
                train_labels_np = train_labels_np.cuda()
                prediction = torch.argmax(torch.bincount(train_labels_np[scores[i]]))

                if prediction == gt_id:
                    correctness = True
                else:
                    correctness = False

                if correctness:
                    correct_cnt += 1

            acc = 100 * correct_cnt / self.num_test

            utils.logger("The accuracy of kNN at K = {} is {}".format(K, acc))

def main():
    train_data, test_data = get_dataset()
    train_loader, test_loader = get_dataloader(train_data, test_data)
    
    model = get_model()
    feature_extractor = get_feature_extractor(model)
    
    # model = get_model_head()
    # feature_extractor = get_feature_extractor_head(model)
    
    # model = get_model_head_advnet()
    # feature_extractor = get_feature_extractor_head_advnet(model)

    runner = KNNRunner(model, feature_extractor, train_loader, test_loader)
    runner.run()

if __name__ == "__main__":
    main()