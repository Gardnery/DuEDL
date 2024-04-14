import argparse
import os
import re
import shutil

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from binary import assd
import importlib
from time import strftime

import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
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
from val_2D import test_single_volume_cct,test_single_volume_ds,test_single_volume_cct_add_noise,test2,test1,test3,test4
from plot import test_single_volume_cct_Uentropy

from itertools import zip_longest



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/server/yyt/WSL4MIS-main/data/ACDC_our', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/pCE_SPS', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name')
parser.add_argument('--fold', type=str,
                    default='test', help='fold')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--sup_type', type=str, default="scribble",
                    help='label')
parser.add_argument('--save_prediction', default='ACDC/pCE_SPS', action="store_true", help='save predictions while testing')
parser.add_argument("--arch", default='ACDC', type=str)

import torch
import torch.nn.functional as F
import math

def transform(t):
    t = t.view(t.size(0), t.size(1), -1)  # [N, C, HW]
    t = t.transpose(1, 2)  # [N, HW, C]
    t = t.contiguous().view(-1, t.size(2))
    return t

def Uentropy(logits, num_classes):
    # 使用 softmax 将 logits 转换为概率分布
    pc = F.softmax(logits, dim=1)  # pc.shape: (batch_size, num_classes, height, width)

    nobackground = pc[:, :, :, :]
    # 计算每个像素位置的熵，但仅考虑非背景类别
    entropy = -torch.sum(nobackground * torch.log(nobackground + 1e-10), dim=1)  # entropy.shape: (batch_size, height, width)

    return entropy
def our_Uentropy(logits, num_classes):
    # 使用 softmax 将 logits 转换为概率分布
    pc = F.softmax(logits, dim=1)  # pc.shape: (batch_size, num_classes, height, width)

    nobackground = pc[:, :, :, :]
    # 计算每个像素位置的熵，但仅考虑非背景类别
    entropy = -(1/math.log(4))*torch.sum(nobackground * torch.log(nobackground + 1e-10), dim=1)  # entropy.shape: (batch_size, height, width)

    return entropy

def calculate_ece(pred_probs, true_labels, n_bins=10):
    ece = 0.0
    pred_probs = pred_probs.squeeze(0)  # 假设batch size为1

    # 获取每个像素点的最高概率值和相应的类别
    max_probs = np.max(pred_probs, axis=0)
    pred_labels = np.argmax(pred_probs, axis=0)

    # 确保true_labels与pred_labels的形状一致
    true_labels = true_labels.reshape(pred_labels.shape)
    
    accuracies = np.zeros(n_bins)
    confidences = np.zeros(n_bins)

    bins = np.linspace(0, 1, n_bins + 1)
    for i in range(len(bins) - 1):
        bin_lower, bin_upper = bins[i], bins[i + 1]

        # 确定每个bin中的索引
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            # 计算准确率：预测类别与真实类别相同的比例
            accuracy_in_bin = np.sum(pred_labels[in_bin] == true_labels[in_bin]) / bin_size
            accuracies[i] = accuracy_in_bin

            # 计算平均置信度
            avg_confidence_in_bin = np.sum(max_probs[in_bin]) / bin_size
            confidences[i] = avg_confidence_in_bin

            # 计算ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * (bin_size / max_probs.size)
    # # 绘制柱状图
    # plt.bar(confidences, accuracies, width=0.1)  # width可以调整柱状图的宽度
    # plt.xlabel('Confidence')  # 横坐标标签
    # plt.ylabel('Accuracy')  # 纵坐标标签
    # plt.title('Accuracy vs Confidence')  # 图表标题

    # # 显示网格
    # plt.grid(True)

    # # 设置纵坐标区间
    # plt.ylim(0., 1.0)  # 您可以根据实际数据的范围来调整区间
    # plt.xlim(0., 1.0)

    # # 绘制斜率为45度的直线
    # x = np.linspace(0, 1, 100)
    # y = x
    # plt.plot(x, y, 'r--')  # 'r--'表示红色虚线，您可以根据需要调整颜色和线型

    # # 保存图表为图片
    # plt.savefig('/home/server/yyt/WSL4MIS-main/code/accuracy_vs_confidenc_origin.png')

    return ece

def softmax_assd_score(output, target):
    ret = []

    # whole (label: 1 ,2 ,3)
    o = output > 0; t = target > 0 # ce
    ret += assd(o, t),
    # core (tumor core 1 and 3)
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    ret += assd(o, t),
    # active (enhanccing tumor region 1 )# 3
    o = (output == 3);t = (target == 3)
    ret += assd(o, t),

    return ret

def computer_ece(probabilities, labels, num_bins=100):
    """
    计算多分类问题的期望校准误差（Expected Calibration Error）。

    参数：
    - probabilities: 模型的概率预测列表，每行包含每个类别的概率
    - labels: 实际标签列表，每个元素为相应样本的真实类别
    - num_bins: 区间的数量

    返回：
    - ece: 期望校准误差
    """
    

    # 将概率和标签转换为 NumPy 数组
    probabilities = np.array(probabilities)
    labels = np.array(labels.flatten())

    # 将每个样本的概率值分成区间
    bin_limits = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(np.max(probabilities, axis=1), bin_limits)

    # 初始化变量
    accuracies = np.zeros(num_bins)
    confidences = np.zeros(num_bins)
    total_samples = len(labels)

    # 计算每个区间内的准确度和平均置信度
    for i in range(1, num_bins + 1):
        bin_mask = bin_indices == i
        bin_labels = labels[bin_mask]
        bin_probabilities = probabilities[bin_mask]

        if len(bin_labels) > 0:
            # 将概率转换为预测的类别
            predicted_classes = np.argmax(bin_probabilities, axis=1)
            bin_accuracy = np.mean(bin_labels == predicted_classes)
            bin_confidence = np.mean(np.max(bin_probabilities, axis=1))
            accuracies[i - 1] = bin_accuracy
            confidences[i - 1] = bin_confidence
    bin_total = np.bincount(bin_indices, minlength=num_bins)
    bin_weight = bin_total / bin_total.sum()
    # 计算 ECE
    ece = np.sum(np.abs(accuracies - confidences) * (np.sum(bin_indices == i) / total_samples) for i in range(1, num_bins + 1))
    return ece.mean()

def computer_sUEO(labels,pred ,output):
    sUEO = []
    uncertainty = our_Uentropy(output, 4)
    uncertainty = transform(uncertainty).cpu().detach().numpy().flatten()
    labels = np.array(labels.flatten())

    # s1 = np.sum(labels * uncertainty)
    # s2 = np.sum(labels * labels + uncertainty * uncertainty)
    ueo = 0
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for threshold in thresholds:
        thresholded_uncertainty = uncertainty > threshold
        corrected_prediction = pred.copy()
        corrected_prediction[thresholded_uncertainty] = pred[thresholded_uncertainty]
        dice = com_dice(corrected_prediction, labels)
        sUEO.append(dice)
    return sum(sUEO)/len(sUEO)

def _check_ndarray(obj):
    if not isinstance(obj, np.ndarray):
        raise ValueError("object of type '{}' must be '{}'".format(type(obj).__name__, np.ndarray.__name__))
import pymia.evaluation.metric as m

def com_dice(prediction,target):
    _check_ndarray(prediction)
    _check_ndarray(target)

    d = m.DiceCoefficient()
    d.confusion_matrix = m.ConfusionMatrix(prediction, target)
    return d.calculate()


def get_fold_ids(fold):
    if fold == "test":
        training_set = ["patient{:0>3}".format(i) for i in
                        [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                         71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90]]
        testing_set = ["patient{:0>3}".format(i) for i in
                          [5, 39, 77, 82, 78, 10, 64, 24, 30, 73, 80, 41, 36, 60, 72,21, 28, 99, 54, 90]]
        return [training_set, testing_set]
    else:
        return "ERROR KEY"


def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    # asd = metric.binary.asd(pred, gt, voxelspacing=spacing)a
    asd = 0
    # hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    hd95 = 0
    return dice, hd95, asd

def save_image(image, path, name,n):
    img = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    pred_image = Image.fromarray(img)
    # 保存为PNG图像
    png_folder_path = path + name
    # 判断文件夹是否存在
    if not os.path.exists(png_folder_path):
        # 如果不存在则创建文件夹
        os.mkdir(png_folder_path)
    png_file_name = f'{n}_{name}.png'
    png_file_path = os.path.join(png_folder_path, png_file_name)
    pred_image.save(png_file_path)


def test_single_volume(case, net, test_save_path, FLAGS,n,noice = 0.0):
    h5f = h5py.File(FLAGS.root_path +
                    "/ACDC_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]

    prediction = np.zeros_like(label)

    ece = []
    sUEO = []
    asd = []

    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        noise = np.random.randn(*slice.shape) * noice
        noisy_image = slice + noise

        # # 保存为PNG图像
        # slice_image = ((noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min()) * 255).astype(np.uint8)
        # slice_image = Image.fromarray(slice_image)    
        # png_folder_path = 'F:\yyt\images'
        # # 判断文件夹是否存在
        # if not os.path.exists(png_folder_path):
        #     # 如果不存在则创建文件夹
        #     os.mkdir(png_folder_path)
        # png_file_name = f'{n}_{ind}image.png'
        # png_file_path = os.path.join(png_folder_path, png_file_name)
        # slice_image.save(png_file_path)

        input = torch.from_numpy(noisy_image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_aux1, out_aux2 = net(input)[0], net(input)[1]
            out_aux1_soft = torch.softmax(out_aux1, dim=1)
            out_aux2_soft = torch.softmax(out_aux2, dim=1)
            out = torch.argmax((out_aux1_soft+out_aux2_soft)*0.5, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)


            # yyt生成mask
            pred1 = ((out - out.min()) / (out.max() - out.min()) * 255).astype(np.uint8)
            pred_image = Image.fromarray(pred1)
            # 保存为PNG图像
            png_folder_path = 'F:\yyt\\result'
            # 判断文件夹是否存在
            if not os.path.exists(png_folder_path):
                # 如果不存在则创建文件夹
                os.mkdir(png_folder_path)
            png_file_name = f'{n}_{ind}pre_mask.png'
            png_file_path = os.path.join(png_folder_path, png_file_name)
            pred_image.save(png_file_path)  

            # 计算ECE
            p = (out_aux1_soft+out_aux2_soft)*0.5
            p = p.cpu().detach().numpy()
            gt = label[ind,:,:].astype(int)
            gt = zoom(
                gt,(256 / x, 256 / y), order=0)

            # # yyt保存原始图象
            # gt_image = ((gt - gt.min()) / (gt.max() - gt.min()) * 255).astype(np.uint8)
            # gt_image = Image.fromarray(gt_image)     
            # # 保存为PNG图像
            # png_folder_path = 'F:\yyt\gt'
            # # 判断文件夹是否存在
            # if not os.path.exists(png_folder_path):
            #     # 如果不存在则创建文件夹
            #     os.mkdir(png_folder_path)
            # png_file_name = f'{n}_{ind}gt.png'
            # png_file_path = os.path.join(png_folder_path, png_file_name)
            # gt_image.save(png_file_path) 

            ece.append(calculate_ece(p, gt))
            sUEO.append(computer_sUEO(gt, out.flatten(),out_aux1+out_aux2))
            asd.append(softmax_assd_score(gt,out))


            prediction[ind] = pred
    case = case.replace(".h5", "")
    org_img_path = r"F:\yyt\WSL4MIS-main\WSL4MIS-main\data\ACDC_training\train\{}.nii.gz".format(case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    if FLAGS.save_prediction:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        img_itk.CopyInformation(org_img_itk)
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        prd_itk.CopyInformation(org_img_itk)
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        lab_itk.CopyInformation(org_img_itk)
        sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    ece_mean = sum(ece)/len(ece)
    sUEO_mean = sum(sUEO)/len(sUEO)
    assd_mean = [(sum(values) / len(values)) for values in zip(*asd)]
    return first_metric, second_metric, third_metric,ece_mean,sUEO_mean,assd_mean

from networks.net_factory import net_factory

def Inference(FLAGS):
    train_ids, test_ids = get_fold_ids(FLAGS.fold)
    all_volumes = os.listdir(
        FLAGS.root_path + "/ACDC_training_volumes")
    image_list = []
    for ids in test_ids:
        new_data_list = list(filter(lambda x: re.match(
            '{}.*'.format(ids), x) != None, all_volumes))
        image_list.extend(new_data_list)
    net = net_factory(net_type= FLAGS.model, in_chns=1, class_num=4)
    checkpoint_path = FLAGS.save_prediction
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint)

    net.eval()


    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    metric_list = 0.0
    ece_total = 0.0
    sUEO_total = 0.0
    assd_total = 0.0
    n = 0
    b = 0
    for case in tqdm(image_list):
        print(case)
        # 打开H5文件
        h5f = h5py.File(FLAGS.root_path +
                        "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]

        label = h5f['label'][:]
        metric_i,ece,sUEO,assd = test3(
            image, label, net, classes=4, path = "/model/ACDC/R-BTrast/test0.05",n = b,T=True,P=True, noise=0.05)
        metric_list += np.array(metric_i)
        ece_total += ece
        sUEO_total += sUEO
        assd_total += sum(assd) / len(assd)
        b += 1
    metric_list = metric_list / len(image_list)
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_ece = ece_total / len(image_list)
    mean_asd = assd_total/len(image_list)
    mean_sUEO = sUEO_total / len(image_list)
    print(performance)
    print(mean_hd95)
    print(mean_ece)
    print(mean_sUEO)
    print(mean_asd)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)
    
