import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import Sampler
import os
from random import shuffle
from PIL import Image

from configs import configs as cfg
import utils
import aug_lib as alib

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __init__ method to drop no-label images
    def __init__(self, root, transform=None):
        super(ImageFolderWithPaths, self).__init__(root, transform=transform)

        original_len = len(self.imgs)
        imgs = []
        samples = []
        targets = []
        for sample_idx in range(original_len):
            pth = self.imgs[sample_idx][0]
            base_name = os.path.basename(pth)

            imgs.append(self.imgs[sample_idx])
            samples.append(self.samples[sample_idx])
            targets.append(self.targets[sample_idx])

        self.imgs = imgs
        self.samples = samples
        self.targets = targets

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]
        data = original_tuple[0]  # --> 3x224x224 --> 7x3x224x224
        label = original_tuple[1]
        if data.shape[0] == 1:
            print('gray images')
            data = torch.cat([data, data, data], dim=0)

        # make a new tuple that includes original and the path
        tuple_with_path = (data, label, path)
        return tuple_with_path

class ImageFolderForNNs(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    - Train: return (query, explanations, model2_target, aug_query), target, query_path
        - query: the query image in tensor format
        - explanations: the NNs of the query image (tensor)
        - model2_target: the label of the query image (tensor)
        - aug_query: the augmented query image (tensor)
        - target: the label of the query image (tensor)
        - query_path: the path of the query image (str)
    - Test:
        - CAR or DOG:
            - return ((query, explanations, model2_target, aug_query), target, query_path)
        - CUB:
            - return ((query, explanations, aug_query), target, query_path)
    """

    def __init__(self, root, train=True, transform=None, nn_dict=None):
        super(ImageFolderForNNs, self).__init__(root, transform=transform)

        self.root = root
        self.train = train
        # Load the pre-computed NNs
        if self.train:
            file_name = cfg.DATA.FAISS_TRAIN_FILE
        else:
            file_name = cfg.DATA.FAISS_TEST_FILE
        
        if nn_dict is not None:
            file_name = nn_dict

        self.faiss_nn_dict = np.load(file_name, allow_pickle=True).item()

        sample_count = len(self.faiss_nn_dict)
        utils.logger(f'Initializing Dataset with NNs read from {file_name} len = {sample_count}')
       

    def __getitem__(self, index):
        query_path, target = self.samples[index]
        base_name = os.path.basename(query_path)

        nns = self.faiss_nn_dict[base_name]['NNs']  # 1NN here
        model2_target = self.faiss_nn_dict[base_name]['label']

        # Transform NNs
        nn = self.loader(nns[0])
        nn = self.transform(nn)

        # Transform query
        sample = self.loader(query_path)
        query = self.transform(sample) 

        return query, nn, model2_target, target, query_path
    
class ImageFolderForNNsExtend(ImageFolderForNNs):
    def __init__(self, root, train=True, transform=None, nn_dict=None):
        super().__init__(root, train=True, transform=transform, nn_dict=nn_dict)
        self.samples = []
        self.imgs = []
        self.targets = []
        
        for key, value in self.faiss_nn_dict.items():
            path = os.path.join(root, value['input_gt'], key)
            gt = self.class_to_idx[value['input_gt']]
            self.samples.append((path, gt))
            self.imgs = self.samples
            self.targets.append(gt)
        
        self.query_transforms = T.Compose([
            T.RandomResizedCrop(224, scale=(0.6, 1), ratio=(0.6, 1)),
            # T.RandomHorizontalFlip(),
            # T.Resize(256),
            # T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        query_path, query_class = self.samples[index]
        base_name = os.path.basename(query_path)

        nns = self.faiss_nn_dict[base_name]['NNs']  # 1NN here
        label = self.faiss_nn_dict[base_name]['label']

        # Transform NNs
        nn = self.loader(nns[0])
        nn = self.transform(nn)
        nn_gt = self.faiss_nn_dict[base_name]['nn_gt']
        nn_gt = self.class_to_idx[nn_gt]
        # Transform query
        sample = self.loader(query_path)
        query = self.query_transforms(sample)

        return query, nn, label, query_class, query_path, nn_gt
    
class ShuffleSampler(Sampler):
    def __init__(self, dataset, batchsize):
        self.dataset = dataset
        self.trunk_size = batchsize
        self.trunk_indices = self._generate_trunk_indices()
        self.indices = self._shuffle_trunks()

    def _generate_trunk_indices(self):
        trunk_indices = []
        for i in range(len(self.dataset) // self.trunk_size):
            indices = list(range(i * self.trunk_size, (i + 1) * self.trunk_size))
            shuffle(indices)
            trunk_indices.append(indices)
        return trunk_indices
    
    def _shuffle_trunks(self):
        shuffled_trunks = np.array(self.trunk_indices)
        np.random.shuffle(shuffled_trunks)
        return list(shuffled_trunks.flatten())

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class ImageFolderForNNsWeight(ImageFolderForNNs):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    - Train: return (query, explanations, model2_target, aug_query, sim), target, query_path
        - query: the query image in tensor format
        - explanations: the NNs of the query image (tensor)
        - model2_target: the label of the query image (tensor)
        - aug_query: the augmented query image (tensor)
        - sim: the similartiies of the query image to its NNs (tensor)
        - target: the label of the query image (tensor)
        - query_path: the path of the query image (str)
    - return (query, nn, model2_target, sim, target, query_path)
    """

    def __init__(self, root, train=True, transform=None, nn_dict=None):
        super(ImageFolderForNNsWeight, self).__init__(root,train, transform, nn_dict)

    def __getitem__(self, index):
        query_path, target = self.samples[index]
        base_name = os.path.basename(query_path)
        sim = self.faiss_nn_dict[base_name]['conf']
        query, nn, model2_target, target, query_path = super().__getitem__(index)
        tuple_with_path = (query, nn, model2_target, sim, target, query_path)
        return tuple_with_path

class DatasetTransforms(object):
    def __init__(self):
        self.train_data_transforms = T.Compose([
            T.RandomApply(torch.nn.ModuleList([T.TrivialAugmentWide()]), p=0.5),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.test_data_transforms =  T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        if cfg.EXP.MODEL == 'vit_base_patch16_384':
            self.train_data_transforms = T.Compose([
                T.RandomApply(torch.nn.ModuleList([T.TrivialAugmentWide()]), p=0.5),
                T.RandomResizedCrop(384),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.test_data_transforms =  T.Compose([
                T.Resize(512),
                T.CenterCrop(384),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        

class BinaryDatasetTransforms(object):
    def __init__(self):
        self.train_data_transforms = T.Compose([
            T.RandomApply(torch.nn.ModuleList([T.TrivialAugmentWide()]), p=0.5),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.test_data_transforms =  T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        if cfg.EXP.MODEL=='vit_base_patch16_384':
            self.train_data_transforms = T.Compose([
                T.RandomApply(torch.nn.ModuleList([T.TrivialAugmentWide()]), p=0.5),
                T.RandomResizedCrop(384),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.test_data_transforms =  T.Compose([
                T.Resize(512),
                T.CenterCrop(384),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
