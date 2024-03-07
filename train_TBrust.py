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
from dataloaders.dataset import BaseDataSets, RandomGenerator, RandomGenerator_1
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from val_2D import test_single_volume_cct, test_single_volume_ds

from itertools import zip_longest

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/server/yyt/WSL4MIS-main/data/ACDC_our', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/ACDC_pCE_double_TruB', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
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

def Dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def softmax_dice(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss torch.nn.BCELoss()
    '''

    loss1 = Dice(output[:, 1, ...], (target[:, 1, ...]).float())
    loss2 = Dice(output[:, 2, ...], (target[:, 2, ...]).float())
    loss3 = Dice(output[:, 3, ...], (target[:, 3, ...]).float())

    return loss1 + loss2 + loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data

def dce_eviloss(p, alpha, c, global_step, annealing_step):
    # L_dice =  TDice(alpha,p,criterion_dl)
    L_dice,_,_,_ = softmax_dice(alpha, p)
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    # digama loss
    L_ace = torch.sum(p * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    # log loss
    # labelK = label * (torch.log(S) -  torch.log(alpha))
    # L_ace = torch.sum(label * (torch.log(S) -  torch.log(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - p) + 1
    L_KL = annealing_coef * KL(alp, c)

    return (L_ace + 0.2 * L_KL)

def tv_loss(predication):
    min_pool_x = nn.functional.max_pool2d(
        predication * -1, (3, 3), 1, 1) * -1
    contour = torch.relu(nn.functional.max_pool2d(
        min_pool_x, (3, 3), 1, 1) - min_pool_x)
    # length
    length = torch.mean(torch.abs(contour))
    return length

def one_hot_encoder(input_tensor):
    tensor_list = []
    for i in range(4):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def KL(alpha, c):
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    beta = torch.ones((1, c)).cuda()
    # Mbeta = torch.ones((alpha.shape[0],c)).cuda()
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

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

def transform(t):
    t = t.view(t.size(0), t.size(1), -1)  # [N, C, HW]
    t = t.transpose(1, 2)  # [N, HW, C]
    t = t.contiguous().view(-1, t.size(2))
    return t

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)
def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), fold=args.fold, sup_type=args.sup_type)

    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)

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
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 1.0
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs, outputs_aux1 = model(
                volume_batch)
            outputs_soft1 = torch.softmax(outputs, dim=1)
            outputs_soft2 = torch.softmax(outputs_aux1, dim=1)

            # 上分值证据
            evidence_main = F.softplus(outputs)
            alpha_main = evidence_main + 1
            alpha_main = transform(alpha_main)
            s_main = torch.sum(alpha_main, dim=1,keepdim = True)

            # 下分值证据
            evidence_aux = F.softplus(outputs_aux1)
            alpha_aux = evidence_aux + 1
            alpha_aux = transform(alpha_aux)
            s_aux = torch.sum(alpha_aux, dim=1,keepdim = True)

            loss_ce1 = ce_loss(outputs, label_batch[:].long())
            loss_ce2 = ce_loss(outputs_aux1, label_batch[:].long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)

            beta = random.random() + 1e-10

            pseudo_supervision = torch.argmax(
                (beta * outputs_soft1.detach() + (1.0-beta) * outputs_soft2.detach()), dim=1, keepdim=False)

            loss_pse_sup = 0.5 * (dice_loss(outputs_soft1, pseudo_supervision.unsqueeze(
                1)) + dice_loss(outputs_soft2, pseudo_supervision.unsqueeze(1)))
            # 计算证据损失
            label = label_batch[:].long().detach()
            label = F.one_hot(label, 5)
            label = label.view(-1,5)

            annealing_coef = torch.min(
                torch.tensor(1.0, dtype=torch.float32),
                torch.tensor(epoch_num / 10, dtype=torch.float32),
            )

            L_ace_main = dce_eviloss(label[:,:4],alpha_main,4,epoch_num,50)
            L_ace_aux = dce_eviloss(label[:,:4],alpha_main,4,epoch_num,50)


            loss_EDL = (L_ace_main + L_ace_aux).mean() / 2


            loss = loss_ce + 0.5 * loss_pse_sup + loss_EDL
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_pse_sup: %f, alpha: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_pse_sup.item(), alpha))


            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                b = 0
                ece_total = 0.0
                sUEO_total = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i, ece, sUEO = test_single_volume_cct(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes,n = b)
                    metric_list += np.array(metric_i)
                    b += 1
                    ece_total += ece
                    sUEO_total += sUEO
                mean_ece = ece_total / len(db_val)
                mean_sUEO = sUEO_total / len(db_val)
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
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num > 0 and iter_num % 500 == 0:
                if alpha > 0.01:
                    alpha = alpha - 0.01
                else:
                    alpha = 0.01

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
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
