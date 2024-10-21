import utils
import torch
import torch.nn as nn
import torchvision
import timm

from configs import configs as cfg

def _load_finernet(model, load_path):
        finernet_dict = torch.load(load_path)['model_state_dict']
        load_dict = dict()
        finernet_keys = finernet_dict.keys()
        model_keys = model.state_dict().keys()

        for finernet_key, model_key in zip(finernet_keys, model_keys):
            if finernet_dict[finernet_key].shape == model.state_dict()[model_key].shape:
                load_dict[model_key] = finernet_dict[finernet_key]

        return load_dict

def _load_feature(model, state_dict):
    for key in ['fc.weight', 'fc.bias', 'classifier.weight', 'classifier.bias', 'head.weight', 'head.bias']:
        if state_dict.get(key) is not None:
            state_dict.pop(key)
    msg = model.load_state_dict(state_dict, strict=False)
    return msg

def Resnet(arch='resnet50', pretrained=None, num_classes=1000):
    '''
    pretrained: `imagenet`, `class`, `inat`, `binary`
    - imagenet: pretrained from imagenet
    - class: pretrained from multi-class classification task
    - inat: pretrained from inaturalist
    - finernet: pretrained from binary classification task
    '''
    if hasattr(torchvision.models, arch):
        model = getattr(torchvision.models, arch)()
        setattr(model, 'fc', nn.Linear(model.fc.in_features, num_classes))
    else:
        raise ValueError(f"Not supported model architecture {arch}")
    
    if pretrained == 'imagenet':
        load_path = 'torchvision'
        state_dict = getattr(torchvision.models, arch)(pretrained=True).state_dict()
    elif pretrained == 'class':
        load_path = cfg.EXP.CLASS.PATH
        state_dict = torch.load(load_path)
    elif pretrained == 'inat':
        load_path = 'pretrained_models/inat2021_supervised_large.pth.tar'
        state_dict = torch.load(load_path)['state_dict']
    elif pretrained == 'advnet':
        load_path = cfg.EXP.ADVNET.PATH
        state_dict = _load_finernet(model, load_path)
    else:
        load_path = None

    if load_path is None:
        utils.logger(f'Initialize {arch} from scratch', level='warning')
    else:
        msg = _load_feature(model, state_dict)
        utils.logger(f'Loaded {load_path} to {arch} with msg {msg}', level="warning")

    return model


def ViT(arch='vit_base_patch16_384', pretrained=None, num_classes=1000):
    """
    arch: 'vit_base_patch16_384', 'vit_base_patch16_224'
    """
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    
    if pretrained == 'imagenet':
        load_path = 'torchvision'
        state_dict = timm.create_model(arch, pretrained=True, num_classes=num_classes).state_dict()
    elif pretrained == 'class':
        load_path = cfg.EXP.CLASS.PATH
        state_dict = torch.load(load_path)
    elif pretrained == 'advnet':
        load_path = cfg.EXP.ADVNET.PATH
        state_dict = _load_finernet(model, load_path)
    else:
        load_path = None

    if load_path is None:
        utils.logger(f'Initialize {arch} from scratch', level='warning')
    else:
        msg = _load_feature(model, state_dict)
        utils.logger(f'Loaded {load_path} to {arch} with msg {msg}', level="warning")

    return model
    

class BinaryMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):  # Set default dropout to 0.2
        super().__init__()
        project_dim = 192
        self.net = nn.Sequential(
            nn.Linear(input_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.GELU(),
            nn.Dropout(dropout),  # Add dropout after first activation
            nn.Linear(project_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Make sure to use `hidden_dim` instead of fixed number
            nn.GELU(),
            nn.Dropout(dropout),  # Add dropout after second activation
            nn.Linear(hidden_dim, 2),  # 2 for binary classification
        )

    def forward(self, x):
        return self.net(x)

class CNN_finernetwork(nn.Module):
    def __init__(self, model, pretrained=None, dropout=0.0, feature_dim=2048):
        super(CNN_finernetwork, self).__init__()

        backbone = Resnet(arch=model, pretrained=pretrained, num_classes=cfg.DATA.NUM_CLASSES)

        conv_features = list(backbone.children())[:-2]  # delete the last fc layer
        self.conv_layers = nn.Sequential(*conv_features)

        self.pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.binary_layer = BinaryMLP(
            2 * feature_dim + 2, 32, dropout=dropout)

        self.agg_branch = nn.Linear(2, 1)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, images, explanations):
        # Process the input images
        input_spatial_feats = self.conv_layers(images)
        input_feat = self.pooling_layer(input_spatial_feats).squeeze()

        # Process the nearest neighbors
        explanations = explanations.squeeze()
        explanation_spatial_feats = self.conv_layers(explanations)

        exp_feat = self.pooling_layer(explanation_spatial_feats).squeeze()

        sep_token = torch.zeros([explanations.shape[0], 1], requires_grad=False).to(input_feat.device)
        exp_feat = exp_feat.to(input_feat.device)
        
        x = self.binary_layer(
            torch.cat([sep_token, input_feat, sep_token, exp_feat], dim=1))

        output3 = self.agg_branch(x)

        output = output3

        return output, input_feat, exp_feat, None

class Transformer_finernetwork(nn.Module):
    def __init__(self, model, pretrained=None, dropout=0.0, feature_dim=768):
        super(Transformer_finernetwork, self).__init__()

        self.backbone = ViT(arch=model, pretrained=pretrained, num_classes=cfg.DATA.NUM_CLASSES)
        feature_dim = self.backbone.head.in_features

        self.binary_layer = BinaryMLP(
            2 * feature_dim + 2, 192, dropout=dropout)

        self.agg_branch = nn.Linear(2, 1)

        # initialize all fc layers to xavier
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, images, explanations):
        # Process the input images
        input_feat = self.backbone.forward_features(images)[:, 0].squeeze()

        # Process the nearest neighbors
        exp_feat = self.backbone.forward_features(explanations)[:, 0].squeeze()
        
        sep_token = torch.zeros([explanations.shape[0], 1], requires_grad=False).to(input_feat.device)
        exp_feat = exp_feat.to(input_feat.device)
        
        x = self.binary_layer(
            torch.cat([sep_token, input_feat, sep_token, exp_feat], dim=1))

        output3 = self.agg_branch(x)

        output = output3

        ## cosine similarity
        input_feat = torch.nn.functional.normalize(input_feat, p=2, dim=1)
        exp_feat = torch.nn.functional.normalize(exp_feat, p=2, dim=1)

        sim = torch.nn.functional.cosine_similarity(input_feat, exp_feat)

        return output, input_feat, exp_feat, sim
    

class ADVNET(nn.Module):
    def __init__(self, model, pretrained=None, dropout=0.0, feature_dim=2048):
        super(ADVNET, self).__init__()

        backbone = Resnet(arch=model, pretrained=pretrained, num_classes=cfg.DATA.NUM_CLASSES)

        conv_features = list(backbone.children())[:-2]  # delete the last fc layer
        self.conv_layers = nn.Sequential(*conv_features)

        prev_dim = backbone.fc.in_features

        self.pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.projection_layer = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(),
            backbone.fc,
        )

    def forward(self, images, explanations):
        # Process the input images
        input_spatial_feats = self.conv_layers(images)
        input_feat = self.pooling_layer(input_spatial_feats).squeeze()

        # Process the nearest neighbors
        explanations = explanations.squeeze()
        explanation_spatial_feats = self.conv_layers(explanations)

        exp_feat = self.pooling_layer(explanation_spatial_feats).squeeze()
        exp_feat = exp_feat.to(input_feat.device)
        
        input_feat = self.projection_layer(input_feat)
        exp_feat = self.projection_layer(exp_feat)
       
        return input_feat, exp_feat


