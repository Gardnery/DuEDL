import torch
import argparse
import os
import torch.nn as nn
#import time
import torch.nn.functional as F
from networks.efficientunet import Effi_UNet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_DS, UNet_CCT, UNet_CCT_3H
from networks.unet_3D import unet_3D
from networks.vnet import VNet
from networks.VoxResNet import VoxResNet
from networks.attention_unet import Attention_UNet
from val_2D import tailor_and_concat
class TMSU(nn.Module):

    def __init__(self, classes, modes, model,input_dims,total_epochs,lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param modes: Number of modes
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMSU, self).__init__()
        # ---- Net Backbone ----
        if model == 'unet_3D' and input_dims =='four':
            self.backbone = unet_3D(in_channels=4, base_channels=16, num_classes=classes)
        elif model == 'attention_unet':
            self.backbone = Attention_UNet(in_channels=1, base_channels=16, num_classes=classes)
        elif model == 'V'and input_dims =='four':
            self.backbone = VNet(n_channels=4, n_classes=classes, n_filters=16, normalization='gn', has_dropout=False)
        elif model == 'V':
            self.backbone = VNet(n_channels=1, n_classes=classes, n_filters=16, normalization='gn', has_dropout=False)
        elif model =='voxresnet':
            self.backbone = VoxResNet(input_dims=input_dims, _conv_repr=True, _pe_type="learned")
        elif model =='unet_cct':
            self.backbone = UNet_CCT(in_chns=in_chns, class_num=classes)
        elif model =='unet_cct_3h':
            self.backbone = UNet_CCT_3H(input_dims=input_dims, _conv_repr=True, _pe_type="learned")        
        elif model == "unet_ds":
            self.backbone = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
        elif model == "efficient_unet":
            self.backbone = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                            in_channels=in_chns, classes=class_num).cuda()
        elif model == "pnet":
            self.backbone = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
        else:
            self.backbone = UNet(in_channels=1, base_channels=16, num_classes=classes)
        self.backbone.cuda()
        self.modes = modes
        self.classes = classes
        self.eps = 1e-10
        self.lambda_epochs = lambda_epochs
        self.total_epochs = total_epochs+1
        # self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.modes)])

    def forward(self, X, y, global_step, mode, use_TTA=False):
        # X data
        # y target
        # global_step : epochs

        # step zero: backbone
        if mode == 'train':
            backbone_output = self.backbone(X)
        elif mode == 'val':
            backbone_output = tailor_and_concat(X, self.backbone)
            # backbone_X = F.softmax(backbone_X,dim=1)
        else:
            if not use_TTA:
                backbone_output = tailor_and_concat(X, self.backbone)
                # backbone_X = F.softmax(backbone_X,dim=1)
            else:
                x = X
                x = x[..., :155]
                logit = F.softmax(tailor_and_concat(x, self.backbone), 1)  # no flip
                logit += F.softmax(tailor_and_concat(x.flip(dims=(2,)), self.backbone).flip(dims=(2,)), 1)  # flip H
                logit += F.softmax(tailor_and_concat(x.flip(dims=(3,)), self.backbone).flip(dims=(3,)), 1)  # flip W
                logit += F.softmax(tailor_and_concat(x.flip(dims=(4,)), self.backbone).flip(dims=(4,)), 1)  # flip D
                logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3)), self.backbone).flip(dims=(2, 3)),
                                   1)  # flip H, W
                logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 4)), self.backbone).flip(dims=(2, 4)),
                                   1)  # flip H, D
                logit += F.softmax(tailor_and_concat(x.flip(dims=(3, 4)), self.backbone).flip(dims=(3, 4)),
                                   1)  # flip W, D
                logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3, 4)), self.backbone).flip(dims=(2, 3, 4)),
                                   1)  # flip H, W, D
                backbone_output = logit / 8.0  # mean
                # backbone_X = F.softmax(backbone_X,dim=1)

        # step one
        evidence = self.infer(backbone_output) # batch_size * class * image_size

        # step two
        alpha = evidence + 1
        if mode == 'train' or mode == 'val':
            loss = dce_eviloss(y.to(torch.int64), alpha, self.classes, global_step, self.lambda_epochs)
            loss = torch.mean(loss)
            return evidence, loss
        else:
            return evidence

    def infer(self, input):
        """
        :param input: modal data
        :return: evidence of modal data
        """
        # evidence = (input-torch.min(input))/(torch.max(input)-torch.min(input))
        evidence = F.softplus(input)
        # evidence[m_num] = torch.exp(torch.clamp(evidence, -10, 10))
        # evidence = F.relu(evidence)
        return evidence

