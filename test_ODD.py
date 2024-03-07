import argparse
import logging
import os
import random
import shutil
import sys
import time
import matplotlib
import h5py

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
from val_2D import test_single_volume_cct,test_single_volume_ds,test_single_volume_cct_add_noise,test2
from itertools import zip_longest

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/server/yyt/WSL4MIS-main/data/MRI', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/test3', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=53800, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)



def transform(t):
    t = t.view(t.size(0), t.size(1), -1)  # [N, C, HW]
    t = t.transpose(1, 2)  # [N, HW, C]
    t = t.contiguous().view(-1, t.size(2))
    return t

def test(args):
    num_classes = args.num_classes  
    batch_size = args.batch_size

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    checkpoint_path = args.exp
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    model.eval()



    folder_path = args.root_path
    metric_list = 0.0
    i = 0
    a = 0
    model.eval()
    b = 0
    metric_list = 0.0
    ece_total = 0.0
    sUEO_total = 0.0

    # 获取文件夹中的所有文件
    file_list = os.listdir(folder_path)
    b = 0
    # 遍历文件夹中的每个文件
    for file_name in file_list:
        if file_name.endswith('.h5'):  # 确保文件是H5文件
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, file_name)

            # 打开H5文件
            h5_file = h5py.File(file_path, 'r')
            image = h5_file['image'][:]

            label = h5_file['label'][:]

            metric_i,ece,sUEO = test2(
                image, label, model, classes=num_classes,n = b)
            metric_list += np.array(metric_i)
            ece_total += ece
            sUEO_total += sUEO
            b += 1
    metric_list = metric_list / len(file_list)
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_ece = ece_total / len(file_list)
    mean_sUEO = sUEO_total / len(file_list)
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
