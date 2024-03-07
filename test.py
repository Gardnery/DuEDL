import argparse
import logging
import os
import random
import shutil
import sys
import time
import matplotlib

from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, RandomGenerator_1
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from val_2D import test_single_volume_cct,test_single_volume_ds,test_single_volume_cct_add_noise,test_single_volume,test_single_volume_add_noise


from itertools import zip_longest

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/server/yyt/WSL4MIS-main/data/ACDC_our', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/test3', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
args = parser.parse_args()





def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)



def test(args):
    num_classes = args.num_classes  
    batch_size = args.batch_size

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    checkpoint_path = args.exp
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    model.eval()


    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")


    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)


    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    metric_list = 0.0
    i = 0
    a = 0
    model.eval()
    b = 0
    metric_list = 0.0
    ece_total = 0.0
    sUEO_total = 0.0
    for i_batch, sampled_batch in enumerate(valloader):
        # metric_i,ece,sUEO = test_single_volume_cct(
        #     sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
        # metric_i,ece,sUEO = test_single_volume_cct_add_noise(
        #    sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, noise = 0.15, n = b)
        # metric_i,ece,sUEO = test_single_volume(
        #    sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, n = b)
        metric_i,ece,sUEO = test_single_volume_add_noise(
           sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, n = b)
        b = b + 1
        metric_list += np.array(metric_i)
        ece_total += ece
        sUEO_total += sUEO
    metric_list = metric_list / len(db_val)
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_ece = ece_total / len(db_val)
    mean_sUEO = sUEO_total / len(db_val)
    print(performance)
    print(mean_hd95)
    print(mean_ece)
    print(mean_sUEO)
    


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    test(args)
