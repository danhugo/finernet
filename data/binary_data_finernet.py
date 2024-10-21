import utils

import faiss
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
import os

from data import binary_data_helpers as bdh
from models import *
from datasets import DatasetTransforms, ImageFolderWithPaths
from configs import configs as cfg

DatasetTransforms = DatasetTransforms() 
torch.backends.cudnn.benchmark = True

BATCH_SIZE = cfg.EXP.ADVNET.BS
NUM_CLASSES = cfg.DATA.NUM_CLASSES

class ModelExtractor():
    def __init__(self) -> None:
        if cfg.EXP.MODEL.__contains__('resnet'):
            self.model = Resnet(arch=cfg.EXP.MODEL, pretrained=cfg.EXP.ADVNET.FEATURE, num_classes=cfg.DATA.NUM_CLASSES)
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1]).cuda()  # feature after avgpool

        if cfg.EXP.MODEL.__contains__('vit'):
            self.model = ViT(arch=cfg.EXP.MODEL, pretrained=cfg.EXP.ADVNET.FEATURE, num_classes=cfg.DATA.NUM_CLASSES)
            self.feature_extractor = self.model

        load_dict = self._load_ckpt()
        self._load_dict(load_dict)

    def _load_ckpt(self):
        path = cfg.EXP.CLASS.PATH
        utils.logger('Load {}'.format(path))
        load_dict = torch.load(path)
        return load_dict
        
    def _load_dict(self, load_dict):
        msg = self.model.load_state_dict(load_dict, strict=False)  
        utils.logger(msg)

class FeatureExtractor():
    def __init__(self, 
            model,
            feature_extractor,
            suffix='_inner_product'
            ) -> None:
        self.suffix = suffix
        self.model = model.cuda().eval()
        self.feature_extractor = feature_extractor.cuda().eval()
        self.in_features = self._get_in_features()
        # self.search_fn = bdh.FaissSearchL2(self.in_features)
        # self.search_all_fn = bdh.FaissSearchL2(self.in_features)
        self.search_fn = bdh.FaissSearchInnerProduct(self.in_features)
        self.search_all_fn = bdh.FaissSearchInnerProduct(self.in_features)
        # self.search_fn = bdh.FaissSearchCosine(self.in_features)

    def _get_in_features(self):
        if cfg.EXP.MODEL.__contains__('resnet'):
            return self.model.fc.in_features
        elif cfg.EXP.MODEL.__contains__('vit'):
            return self.model.embed_dim

    def _get_knowledge_base_dataset(self):
        dataset = datasets.ImageFolder(
            root=cfg.DATA.TRAIN_PATH,
            transform=DatasetTransforms.train_data_transforms
        )

        return dataset
    
    def _get_class_subset_dict(self, dataset):
        utils.logger('Getting class subset dict ...')
        class_subset_dict = dict()
        targets = dataset.targets
        for class_id in tqdm(range(len(dataset.class_to_idx))):
            class_list = [x for x in range(len(targets)) if targets[x] == class_id] # index of all samples having the same class id
            class_subset = Subset(dataset, class_list)
            class_subset_dict[class_id] = class_subset
        
        return class_subset_dict

    def _get_embedding(self, dataset, class_subset_dict):
        # Create `faiss_embedding_class_dict` dict of {class_0: faiss_class_0_index} for KNN retrieval
        utils.logger("Building FAISS index...! Training set is the knowledge base.")

        faiss_classes_index_dict = dict()
        data_embeddings = []
        for class_id in tqdm(range(len(dataset.class_to_idx))):
            class_subset = class_subset_dict[class_id]
            class_loader = DataLoader(class_subset, batch_size=cfg.EXP.ADVNET.BS, shuffle=False, num_workers=8, pin_memory=True)
            class_embeddings = []
            for data, _ in class_loader:
                if cfg.EXP.MODEL.__contains__('resnet'):
                    embeddings = self.feature_extractor(data.cuda())  # 512x1 for RN18 2048 for RN50
                if cfg.EXP.MODEL.__contains__('vit'):
                    embeddings = self.feature_extractor.forward_features(data.cuda())[:, 0]
                embeddings = torch.flatten(embeddings, start_dim=1)
                class_embeddings.append(embeddings.cpu().detach().numpy())
                data_embeddings.append(embeddings.cpu().detach().numpy())
            class_embeddings = np.vstack(class_embeddings)

            self.search_fn.fit(class_embeddings)
            faiss_classes_index_dict[class_id] = self.search_fn.index
            class_subset_dict[class_id] = class_subset
            self.search_fn.reset()

        data_embeddings = np.vstack(data_embeddings)
        self.search_all_fn.fit(data_embeddings)
     
        return faiss_classes_index_dict
    
    def _get_extract_config(self, datatype):
        if datatype == 'train':
            base_name = 'nns_top{}_{}{}'.format(cfg.EXP.ADVNET.NEG_PRED, cfg.EXP.MODEL, self.suffix)
            data_path = cfg.DATA.TRAIN_PATH
            data_transform = DatasetTransforms.train_data_transforms
            depth_of_pred = cfg.EXP.ADVNET.NEG_PRED
            file_name = 'faiss/{}/train_{}.npy'.format(cfg.DATA.DATASET, base_name)
            nn_data_path = f'{data_path}_{base_name}'
            
        elif datatype == 'test':
            data_path = cfg.DATA.TEST_PATH
            data_transform = DatasetTransforms.test_data_transforms
            depth_of_pred = 1
            file_name = 'faiss/{}/test_nns_{}{}.npy'.format(cfg.DATA.DATASET, cfg.EXP.MODEL, self.suffix)
            nn_data_path = None

        extract_config = {
            'datatype': datatype,
            'data_path': data_path,
            'nn_data_path': nn_data_path,
            'data_transform': data_transform,
            'depth_of_pred': depth_of_pred,
            'file_name': file_name,
            'k_value': cfg.EXP.ADVNET.SAMPLE_NEG,
        }
        return extract_config

    def _get_extract_loader(self, config):
        dataset = ImageFolderWithPaths(config['data_path'], config['data_transform'])
        utils.logger(f"Getting contrastive dataset from: {config['datatype']} set, data path: {config['data_path']} ...")

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        return loader

    def _write_faiss_nn_dict(self, faiss_nn_dict, key, nn_list, label, conf, input_gt, input_path):
        faiss_nn_dict[key] = dict()
        faiss_nn_dict[key]['NNs'] = nn_list
        faiss_nn_dict[key]['label'] = label
        faiss_nn_dict[key]['conf'] = conf
        faiss_nn_dict[key]['input_gt'] = input_gt
        faiss_nn_dict[key]['input_path'] = input_path
    
    def _get_nns_sample(self, batch_data, label, data_paths, class_subset_dict, faiss_classes_index_dict, faiss_nn_dict, config):
        depth_of_pred = config['depth_of_pred']
        k_value = config['k_value']

        if cfg.EXP.MODEL.__contains__('resnet'):
            embeddings = self.feature_extractor(batch_data.cuda())
        if cfg.EXP.MODEL.__contains__('vit'):
            embeddings = self.feature_extractor.forward_features(batch_data.cuda())[:, 0]

        embeddings = torch.flatten(embeddings, start_dim=1)
        embeddings = embeddings.cpu().detach().numpy()

        out = self.model(batch_data.cuda())
        class_poss = torch.nn.functional.softmax(out, dim=1)
        score, index = torch.topk(class_poss, depth_of_pred, dim=1) # index: batch_size x num_classes, scores: confidence sample x in class c
        for sample_idx in range(batch_data.shape[0]):
            base_name = os.path.basename(data_paths[sample_idx])
            gt_id = label[sample_idx]
            max_dist = self.search_all_fn.search(embeddings[sample_idx].reshape([1, self.in_features]), self.search_all_fn.index.ntotal)[0][0][-1]

            for i in range(depth_of_pred):
                # Get predicted class id in top-k predictions
                predicted_idx = index[sample_idx][i].item()

                # Dataloader and knowledge base upon the predicted class
                prediction_subset = class_subset_dict[predicted_idx]
                faiss_prediction_index = faiss_classes_index_dict[predicted_idx]
                self.search_fn.add_index(faiss_prediction_index)
                nn_list = list()

                if depth_of_pred == 1:  # For val and test sets
                    
                    dists, indices = self.search_fn.search(embeddings[sample_idx].reshape([1, self.in_features]), 1)

                    indices = indices[:, :]
                    dists = dists[:, :]
                    weight = 1 - dists[0, 0] / max_dist # similarity = 1 / (1 + l2_distance): no need with cosine sims

                    for id in range(indices.shape[1]):
                        id = prediction_subset.indices[indices[0, id]]
                        nn_list.append(prediction_subset.dataset.imgs[id][0])

                    key = base_name
                    self._write_faiss_nn_dict(
                        faiss_nn_dict=faiss_nn_dict,
                        key=base_name,
                        nn_list=nn_list,
                        label=int(predicted_idx == gt_id),
                        conf=weight,
                        input_gt=prediction_subset.dataset.classes[gt_id.item()],
                        input_path=data_paths[sample_idx],
                    )
                else:
                    if i == 0:
                        dists, indices = self.search_fn.search(embeddings[sample_idx].reshape([1, self.in_features]), self.search_fn.index.ntotal)
                        
                        for j in range(depth_of_pred):  # Make up x NN sets from top-1 predictions
                            weight = 1 - dists[0, j] / max_dist
                            nn_list = list()
                            if predicted_idx == gt_id:
                                key = 'Correct_{}_{}_'.format(i, j) + base_name
                                min_id = (j * k_value) + 1  # Start from NN1 since NN0 is the input itself
                                max_id = ((j * k_value) + k_value) + 1
                            else:
                                key = 'Wrong_{}_{}_'.format(i, j) + base_name
                                # Start from NN0 of 1st prediction since 1st prediction is wrong
                                min_id = j * k_value  
                                max_id = (j * k_value) + k_value

                            for id in range(min_id, max_id):
                                id = prediction_subset.indices[indices[0, id]]
                                nn_list.append(prediction_subset.dataset.imgs[id][0])

                            self._write_faiss_nn_dict(
                                faiss_nn_dict=faiss_nn_dict,
                                key=key,
                                nn_list=nn_list,
                                label=int(predicted_idx == gt_id),
                                conf=weight,
                                input_gt=prediction_subset.dataset.classes[gt_id.item()],
                                input_path=data_paths[sample_idx],
                            )
                    else:
                        if predicted_idx == gt_id:
                            key = 'Correct_{}_'.format(i) + base_name
                            dists, indices = self.search_fn.search(embeddings[sample_idx].reshape([1, self.in_features]), k_value + 1)
                            indices = indices[:, 1:]  # skip the 1st NN since it is the input itself
                        else:
                            key = 'Wrong_{}_'.format(i) + base_name
                            dists, indices = self.search_fn.search(embeddings[sample_idx].reshape([1, self.in_features]), 1)

                        indices = indices[:, :]
                        weight = 1 - dists[0, 0] / max_dist

                        for id in range(indices.shape[1]):
                            id = prediction_subset.indices[indices[0, id]]
                            nn_list.append(prediction_subset.dataset.imgs[id][0])

                        self._write_faiss_nn_dict(
                            faiss_nn_dict=faiss_nn_dict,
                            key=key,
                            nn_list=nn_list,
                            label=int(predicted_idx == gt_id),
                            conf=weight,
                            input_gt=prediction_subset.dataset.classes[gt_id.item()],
                            input_path=data_paths[sample_idx],
                        )

    def _get_contrastive_data(self, config, class_subset_dict, faiss_classes_index_dict):
        """
        Create faiss_nn_dict (dict()) contains (key, value):
        - 'NNs': list NNs for each top-k prediction
        - 'label': 1 if the prediction is correct, 0 otherwise
        - 'conf': confidence score of the prediction
        - 'input_gt': the input image tensor
        """

        loader = self._get_extract_loader(config)
        faiss_nn_dict = dict()

        for batch_data, label, data_paths in tqdm(loader):
            self._get_nns_sample(batch_data, label, data_paths, class_subset_dict, faiss_classes_index_dict, faiss_nn_dict, config)

        utils.logger(f"length of faiss_nn_dict {len(faiss_nn_dict)} store in file {config['file_name']}")

        bdh.faiss_nn_sanity_check(faiss_nn_dict)
        faiss_nn_dict = bdh.faiss_nn_clearn_duplicate(faiss_nn_dict)

        np.save(config['file_name'], faiss_nn_dict)
    
    def run(self):
        base_dataset = self._get_knowledge_base_dataset()
        class_subset_dict = self._get_class_subset_dict(base_dataset)
        faiss_classes_index_dict = self._get_embedding(base_dataset, class_subset_dict)
        
        for datatype in ['train', 'test']:
            config = self._get_extract_config(datatype)
            self._get_contrastive_data(config, class_subset_dict, faiss_classes_index_dict)
            if datatype == 'train':
                bdh.copy_binary_data(src_path=config['data_path'], dest_path=config['nn_data_path'], nn_path=config['file_name'])

if __name__ == '__main__':
    model_extractor = ModelExtractor()
    feature_extractor = FeatureExtractor(
        model=model_extractor.model, 
        feature_extractor=model_extractor.feature_extractor,
    )
    feature_extractor.run()