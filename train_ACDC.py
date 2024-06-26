import argparse
import logging
import os
import random
import shutil
import sys
import time

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
from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from val_2D import test_single_volume_cct, test_single_volume_ds
from dataloaders.dataset_scribblevc import BaseDataSets, RandomGenerator, Zoom, ACDCDataSets, MSCMRDataSets

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/server/yyt/WSL4MIS-main/data/ACDC_our', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/pCE_SPS', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_epoches', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--d', type=str,
                    default='EDL', help='cross validation')
args = parser.parse_args()

def evident_fusion(b1,u1,b2,u2):
    # b^0 @ b^(0+1)
    bb = torch.bmm(b1.view(-1, 4, 1), b2.view(-1, 1, 4))
    # b^0 * u^1
    uv1_expand = u2.expand(b1.shape)
    bu = torch.mul(b1, uv1_expand)
    # b^1 * u^0
    uv_expand = u1.expand(b1.shape)
    ub = torch.mul(b2, uv_expand)
    # calculate K
    bb_sum = torch.sum(bb, dim=(1, 2), out=None)
    bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
    K = bb_sum - bb_diag

    # calculate b^a
    b_a = (torch.mul(b1, b2) + bu + ub) / ((1 - K).view(-1, 1).expand(b1.shape))
    # calculate u^a
    u_a = torch.mul(u1, u2) / ((1 - K).view(-1, 1).expand(u1.shape))
    # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors
    
    S_a = 4 / u_a
    # calculate new e_k
    e_a = torch.mul(b_a, S_a.expand(b_a.shape))
    alpha_a = e_a + 1
    # calculate new e_k
    # e_a = (4 * b_a) / u_a.expand(a1.shape)
    # # calculate new S
    # S_a = torch.sum(e_a + 1, dim = 1)
    # alpha_a = e_a + 1
    return S_a,e_a,alpha_a,u_a,b_a

def evident_fusion_1(b1,u1,b2,u2):
    C_main = torch.zeros(b1.size())
    # S = torch.sum(b2, dim = 1)
    # for i in range(4):
    #     C_main[:, i, :, :] = b1[:, i, :, :] + S
    C_main = (b1[:, 0, :, :]*b2[:, 1, :, :] + b1[:, 0, :, :]*b2[:, 2, :, :] + b1[:, 0, :, :]*b2[:, 3, :, :] + b1[:, 1, :, :]*b2[:, 0, :, :] + b1[:, 1, :, :]*b2[:, 2, :, :] + b1[:, 1, :, :]* b2[:, 3, :, :] + b1[:, 2, :, :]*b2[:, 3, :, :] + b1[:, 2, :, :]*b2[:, 0, :, :] + b1[:, 2, :, :]*b2[:, 1, :, :] + b1[:, 3, :, :]*b2[:, 0, :, :] + b1[:, 3, :, :]*b2[:, 1, :, :] + b1[:, 3, :, :]*b2[:, 2, :, :])
    b = torch.zeros(b1.size()).cuda()
    u = torch.zeros(u1.size()).cuda()
    for i in range(4):
        b[:,i,:,:] = 1/(1 - C_main+ 1e-8) * (b1[:,i,:,:] * b2[:, i, :, :] + b1[:, i, :, :] * u2 + b2[:, i, :, :] * u1)
    u = 1/(1 - C_main + 1e-8) * (u1 * u2)
    S_a = 4 / (u + 1e-8)
    e_a = S_a * b
    alpha_a = e_a + 1
    return S_a,e_a,alpha_a,u,b


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True) + 1e-8
    gamma1_alp = torch.polygamma(1, alpha)
    gamma1_alp0 = torch.polygamma(1, S)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood



def tv_loss(predication):
    min_pool_x = nn.functional.max_pool2d(
        predication * -1, (3, 3), 1, 1) * -1
    contour = torch.relu(nn.functional.max_pool2d(
        min_pool_x, (3, 3), 1, 1) - min_pool_x)
    # length
    length = torch.mean(torch.abs(contour))
    return length


def transform(t):
    t = t.view(t.size(0), t.size(1), -1)  # [N, C, HW]
    t = t.transpose(1, 2)  # [N, HW, C]
    t = t.contiguous().view(-1, t.size(2))
    return t

def one_hot_encoder(input_tensor):
    tensor_list = []
    for i in range(4):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    alpha = alpha + 1e-8
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def tv_loss(predication):
    min_pool_x = nn.functional.max_pool2d(
        predication * -1, (3, 3), 1, 1) * -1
    contour = torch.relu(nn.functional.max_pool2d(
        min_pool_x, (3, 3), 1, 1) - min_pool_x)
    # length
    length = torch.mean(torch.abs(contour))
    return length


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    # max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    db_train = ACDCDataSets(base_dir=args.root_path, split="train", transform=
        transforms.Compose([RandomGenerator(args.patch_size)]), fold=args.d, sup_type=args.sup_type)

    db_val = ACDCDataSets(base_dir=args.root_path, fold=args.d, split="val")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    batch_size = args.batch_size

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_iterations = args.max_epoches * len(trainloader)
    best_performance = 0.0
    for epoch_num in range(args.max_epoches):
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs, outputs_aux1 = model(
                volume_batch)


             # 上分值证据
            evidence_main = F.softplus(outputs)
            alpha_main = evidence_main + 1
            s_main = torch.sum(evidence_main, dim=1,keepdim = True)
            p_main = alpha_main /s_main

            # 下分值证据
            evidence_aux = F.softplus(outputs_aux1)
            alpha_aux = evidence_aux + 1
            s_aux = torch.sum(evidence_aux, dim=1,keepdim = True)
            p_aux = alpha_aux /s_aux

            f_S, _, f_alpha, _, _ = evident_fusion_1(evidence_main/s_main, 4/s_main, evidence_aux/s_aux, 4/s_aux)
            f_p = f_alpha / f_S

            pseudo_supervision = torch.argmax(f_p.detach(), dim = 1, keepdim = False)

            loss_ce1 = ce_loss(outputs, label_batch[:].long())
            loss_ce2 = ce_loss(outputs_aux1, label_batch[:].long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)



            loss_pse_sup = 0.5 * (dice_loss(p_main, pseudo_supervision.unsqueeze(
                1)) + dice_loss(p_aux, pseudo_supervision.unsqueeze(1)))

            label = label_batch[:].long().detach()
            label = F.one_hot(label, 5)
            label = label.view(-1,5)

            f_alpha = transform(f_alpha)
            alpha_main = transform(alpha_main)
            alpha_aux = transform(alpha_aux)

            
            annealing_coef = torch.min(
                torch.tensor(1.0, dtype=torch.float32),
                torch.tensor(epoch_num / 10, dtype=torch.float32),
            )
            
            loss1 = loglikelihood_loss(label[:,:4], f_alpha,device="cuda") + annealing_coef * kl_divergence((f_alpha - 1) * (1 - label[:,:4]) + 1, 4, device="cuda")
            loss2 = loglikelihood_loss(label[:,:4], alpha_main,device="cuda") + annealing_coef * kl_divergence((alpha_main - 1) * (1 - label[:,:4]) + 1, 4, device="cuda")
            loss3 = loglikelihood_loss(label[:,:4], alpha_aux,device="cuda") + annealing_coef * kl_divergence((alpha_aux - 1) * (1 - label[:,:4]) + 1, 4, device="cuda")


            loss_EDL = (loss1 + loss2 + loss3) * (1 - label[:,4].view(-1,1)) / 3
            loss_EDL = loss_EDL.sum()/(1 - label[:,4].view(-1,1)).sum()


            loss =  0.3 * loss_pse_sup + loss_EDL
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

        if (epoch_num + 1) % 1 == 0:
            model.eval()
            metric_list = 0.0
            for i_batch, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume_cct(
                    sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                metric_list += np.array(metric_i)
            metric_list = metric_list / len(db_val)
            for class_i in range(num_classes-1):
                writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                metric_list[class_i, 0], iter_num)
                writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                metric_list[class_i, 1], iter_num)

            performance = np.mean(metric_list, axis=0)[0]

            mean_hd95 = np.mean(metric_list, axis=0)[1]
            writer.add_scalar('info/val_mean_dice', performance, iter_num)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

            if performance > best_performance:
                best_performance = performance
                save_mode_path = os.path.join(snapshot_path,
                                            'epoch_num_{}_dice_{}.pth'.format(
                                                epoch_num, round(best_performance, 4)))
                save_best = os.path.join(snapshot_path,
                                        '{}_best_model.pth'.format(args.model))
                torch.save(model.state_dict(), save_mode_path)
                torch.save(model.state_dict(), save_best)

            logging.info(
                'epoch %d : mean_dice : %f mean_hd95 : %f' % (epoch_num, performance, mean_hd95))
            model.train()

    writer.close()
    return "Training Finished!"


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

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
