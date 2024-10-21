# Enhancing Multi-Class Fine-Grained Classification through Hard Sample Discrimination


In fine-grained classification, models often misclassify visually similar samples from different classes. This raises the question: *if a model can better distinguish between these hard-to-discriminate samples, can it improve overall performance in multi-class classification tasks?* Based on this hypothesis, we propose a novel method for optimizing multi-class fine-grained classification by training the model to better differentiate visually similar, difficult samples.

## Dataset
- Download datasets: [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/), Stanford Cars, Stanford Dogs, [Flower 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- Place datasets in `data` directory, organized with the structure `data_name/train`, `data_name/test`

## Installation
### Environments
```
python 3.10
torch 2.2.2+cu11.8
```
### Install with conda
```
conda env create -f environment.yml
```
## Train

### Training Setup
Modify configs.py to allow selection of dataset and feature architecture settings. For example
```python
# configs.merge_from_file("experiments/resnet50_car.yaml")
configs.merge_from_file("experiments/resnet50_cub.yaml")
```
### Training Pipeline
```
1. Fine-tune model on multi-class classification task.
2. Leverage the trained model from phase 1 to obtain pairs of visually similar samples using the k-Nearest Neighbors (kNN) algorithm.
3. Fine-tune the model on a binary classification task aimed at determining whether two samples belong to the same class or not.
4. Use pretrained model from phase 3 and retrain again on the multi-class classification task.
```
### Run the training
1. Fine-tune model on multi-class classification task.
```
python src/linear_probe.py
```
2. Constructing the dataset for Hard Sample Discrimination.
```
python data/binary_data_finernet.py
```
3. Fine-tune model on binary-class classification task.
```
python src/model_training.py
```
4. Retrain the model for the multi-class classification task using the weights initialized from the trained model in Phase 3.
```
python linear_probe.py
```

## Experiments & Results
Model accuracy with single-step training (Step 1) and with the FineNet training pipeline.
### Resnet-50 arch

| Pipeline                  | Cub | Stanford Cars | Stanford Dogs | Flowers 102| DTD |
|:--------------------------|:---:|:-------------:|:-------------:|:----------:|:---:|
| single-step finetune      | 87.90 | 90.78 | 88.61 | 97.12 | 77.23 |
| finernet                  | 88.44 | 91.62 | 88.72 | 97.53 | 77.55 |

### ViT-B-16 arch

| Pipeline                  | Cub | Stanford Cars | Stanford Dogs | Flowers 102| DTD |
|:--------------------------|:---:|:-------------:|:-------------:|:----------:|:---:|
| single-step finetune      | 90.39 | 92.20 | 89.09 | - | 83.19 |
| finernet                  | 90.66 | 93.38 | 89.38 | - | 84.41 |
