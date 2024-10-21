import faiss
import numpy as np
import torch
import os
import shutil

import utils
from configs import configs as cfg

def faiss_nn_sanity_check(faiss_nn_dict):
    cnt = 0
    for _, v in faiss_nn_dict.items():
        NN_class = os.path.basename(os.path.dirname(v['NNs'][0]))
        input_class = os.path.basename(os.path.dirname(v['input_path']))
        if v['label'] == 1:
            cnt += 1
            if NN_class.lower() == input_class.lower():
                continue
            else:
                utils.logger('You retrieved wrong NNs. The label for the pair is positive but two images are from different classes!', level='error')
                exit(-1)
        else:
            if NN_class.lower() != input_class.lower():
                continue
            else:
                utils.logger('You retrieved wrong NNs. The label for the pair is negative but two images are from same class!', level='error')
                exit(-1)

    if cnt / len(faiss_nn_dict) > 0.7 or cnt / len(faiss_nn_dict) < 0.3:
        utils.logger('The ratio of positive samples is skewed {:.4f}'.format(cnt / len(faiss_nn_dict)), level='warning')

    utils.logger('Passed sanity checks for extracting NNs!')

def faiss_nn_clearn_duplicate(faiss_nn_dict):
    new_dict = dict()
    for k, v in faiss_nn_dict.items():
        for nn in v['NNs']:
            base_name = os.path.basename(nn)
            if base_name in k:
                break
            else:
                new_dict[k] = v
    utils.logger('DONE: Cleaning duplicates entries: query and NN being similar!')
    return new_dict

def copy_to_augment_folder(source_folder, destination_folder):
    if os.path.exists(destination_folder):
        utils.logger(f"Destination folder '{destination_folder}' already exists. Delete it.")
        shutil.rmtree(destination_folder)

    shutil.copytree(source_folder, destination_folder, ignore=shutil.ignore_patterns(f'*.jpg'))
    utils.logger(f"Copied  source folder: '{source_folder}' to  augment folder: '{destination_folder}'")

def copy_binary_data(src_path, dest_path, nn_path):
    # Set the paths and filenames
    source_folder = src_path
    folder_path = dest_path
    dict_path = nn_path
    
    copy_to_augment_folder(source_folder, folder_path)
    
    # Load the dictionary
    file_dict = np.load(dict_path, allow_pickle=True).item()
    cnt = 0

    for file_key, value in file_dict.items():
        src_path = value['input_path']
        parent_dir = os.path.basename(os.path.dirname(src_path))
        dst_path = os.path.join(folder_path, parent_dir, file_key)
        shutil.copyfile(src_path, dst_path)
        cnt += 1

    utils.logger(f"{cnt} files copied successfully!")

def search_cosine_faiss(embeddings:np.ndarray, query:np.ndarray, k:int):
    """nearest neighbor search using faiss"""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    faiss.normalize_L2(query)
    index.add(embeddings)
    D, I = index.search(query, k)
    return D, I

def search_cosine_torch(embeddings:torch.tensor, query:torch.tensor, k:int):
    """nearest neighbor search using pytorch"""
    cosine_similarities = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(0), query.unsqueeze(1), dim=2)
    sorted_similarities, sorted_indices = torch.topk(cosine_similarities, k=k, dim=1, sorted=True)
    return sorted_similarities, sorted_indices


class FaissSearchL2():
    def __init__(self, in_features:int):
        self.in_features = in_features
        self.index = faiss.IndexFlatL2(self.in_features)
        
    def fit(self, embeddings:np.ndarray):
        self.index.add(embeddings)
    
    def add_index(self, index: faiss.IndexFlatL2):
        self.index = index

    def search(self, query:np.ndarray, k:int):
        D, I = self.index.search(query, k)
        return D, I
    
    def reset(self):
        self.index = faiss.IndexFlatL2(self.in_features)
    
class FaissSearchCosine():
    def __init__(self, in_features:int):
        self.in_features = in_features
        self.index = faiss.IndexFlatIP(self.in_features)

    def fit(self, embeddings:np.ndarray):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def add_index(self, index: faiss.IndexFlatIP):
        self.index = index

    def search(self, query:np.ndarray, k:int):
        faiss.normalize_L2(query)
        D, I = self.index.search(query, k)
        return D, I
    
    def reset(self):
        self.index = faiss.IndexFlatIP(self.in_features)

class FaissSearchInnerProduct():
    def __init__(self, in_features:int):
        self.in_features = in_features
        self.index = faiss.IndexFlatIP(self.in_features)
        
    def fit(self, embeddings:np.ndarray):
        self.index.add(embeddings)
    
    def add_index(self, index: faiss.IndexFlatIP):
        self.index = index

    def search(self, query:np.ndarray, k:int):
        D, I = self.index.search(query, k)
        return D, I
    
    def reset(self):
        self.index = faiss.IndexFlatIP(self.in_features)

