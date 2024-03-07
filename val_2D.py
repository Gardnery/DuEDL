import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import pymia.evaluation.metric as m
from torch import nn


def _check_ndarray(obj):
    if not isinstance(obj, np.ndarray):
        raise ValueError("object of type '{}' must be '{}'".format(type(obj).__name__, np.ndarray.__name__))

def transform(t):
    t = t.view(t.size(0), t.size(1), -1)  # [N, C, HW]
    t = t.transpose(1, 2)  # [N, HW, C]
    t = t.contiguous().view(-1, t.size(2))
    return t

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1


    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        if dice == 0.0:
            hd95 = 0.0
        else:
            hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0

def calculate_dice(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    else:
        return 0
    
import torch
import torch.nn.functional as F
import math

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

def com_dice(prediction,target):
    _check_ndarray(prediction)
    _check_ndarray(target)

    d = m.DiceCoefficient()
    d.confusion_matrix = m.ConfusionMatrix(prediction, target)
    return d.calculate()
    

def colormap_fn(label):
    # 这里根据不同类别的标签返回不同的颜色，比如背景为黑色，其他类别可以是不同的颜色
    if label == 0:  # 背景
        return (0, 0, 0)  # 黑色
    elif label == 1:  # 类别1
        return (255, 0, 0)  # 红色
    elif label == 2:  # 类别2
        return (0, 255, 0)  # 绿色
    elif label == 3:  # 类别3
        return (0, 0, 255)  # 蓝色
    # 添加更多类别的映射


def test_single_volume(image, label, net, classes, patch_size=[256, 256],n = 0, epoch = 0):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    ece = []
    sUEO = []
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()

            gt = label[ind,:,:].astype(int)
            gt = zoom(
                gt, (patch_size[0] / x, patch_size[1] / y), order=0)

            net.eval()
            with torch.no_grad():
                output_main = net(input)
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)

                # 计算ECE
                p = torch.softmax(output_main, dim=1)
                p = transform(p).cpu().detach().numpy()
                ece.append(computer_ece(p, gt))
                sUEO.append(computer_sUEO(gt, out.flatten(),output_main))                

                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            # 计算ECE
            p = torch.softmax(output_main, dim=1)
            p = transform(p).cpu().detach().numpy()
            ece.append(computer_ece(p, gt))
            sUEO.append(computer_sUEO(gt, prediction.flatten(),output_main))
    metric_list = []
    ece_mean = sum(ece)/len(ece)
    sUEO_mean = sum(sUEO)/len(sUEO)
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list,ece_mean,sUEO_mean


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output_main, _, _, _ = net(input)
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

def test2(image, label, net, classes, patch_size=[256, 256],n = 0, epoch = 0):
    # image, label = image.squeeze(0).cpu().detach(
    # ).numpy(), label.squeeze(0).cpu().detach().numpy()
    ece = []
    sUEO = []
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                 0).unsqueeze(0).float().cuda()
            gt = label[ind,:,:].astype(int)
            gt = zoom(
                gt, (patch_size[0] / x, patch_size[1] / y), order=0)

            net.eval()
            with torch.no_grad():
                output_main = net(input)[0]
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                # 计算ECE
                p = torch.softmax(output_main, dim=1)
                p = transform(p).cpu().detach().numpy()
                ece.append(computer_ece(p, gt))
                sUEO.append(computer_sUEO(gt, out.flatten(),output_main))
                # print(,label))
                prediction[ind] = pred

    else:
        # input = torch.from_numpy(image).unsqueeze(
        #     0).unsqueeze(0).float().cuda()
        x, y = image.shape[0], image.shape[1]
        image = zoom(
            image, (patch_size[0] / x, patch_size[1] / y), order=0)

        # yyt保存原始图象
        slice_image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        slice_image = Image.fromarray(slice_image)     
        # # 保存为PNG图像
        # png_folder_path = '/home/server/yyt/model/image'
        # # 判断文件夹是否存在
        # if not os.path.exists(png_folder_path):
        #     # 如果不存在则创建文件夹
        #     os.mkdir(png_folder_path)
        # png_file_name = f'{n}_image.png'
        # png_file_path = os.path.join(png_folder_path, png_file_name)
        # slice_image.save(png_file_path) 

        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        gt = label.astype(int)
        gt = zoom(
            gt, (patch_size[0] / x, patch_size[1] / y), order=0)


        # # yyt保存原始图象
        # gt_image = ((gt - gt.min()) / (gt.max() - gt.min()) * 255).astype(np.uint8)
        # gt_image = Image.fromarray(gt_image)     
        # # 保存为PNG图像
        # png_folder_path = '/home/server/yyt/model/gt'
        # # 判断文件夹是否存在
        # if not os.path.exists(png_folder_path):
        #     # 如果不存在则创建文件夹
        #     os.mkdir(png_folder_path)
        # png_file_name = f'{n}_gt.png'
        # png_file_path = os.path.join(png_folder_path, png_file_name)
        # gt_image.save(png_file_path) 
        

        # yyt保存原始图象
        label = ((gt - gt.min()) / (gt.max() - gt.min()) * 255).astype(np.uint8)
        label = Image.fromarray(label)       

        net.eval()
        with torch.no_grad():
            output_main = net(input)[0]
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

            # yyt生成mask
            pred = ((prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255).astype(np.uint8)
            pred_image = Image.fromarray(pred)
            # # 保存为PNG图像
            # png_folder_path = '/home/server/yyt/model/result'
            # # 判断文件夹是否存在
            # if not os.path.exists(png_folder_path):
            #     # 如果不存在则创建文件夹
            #     os.mkdir(png_folder_path)
            # png_file_name = f'{n}_pre_mask.png'
            # png_file_path = os.path.join(png_folder_path, png_file_name)
            # pred_image.save(png_file_path)

            # 计算ECE
            p = torch.softmax(output_main, dim=1)
            p = transform(p).cpu().detach().numpy()
            ece.append(computer_ece(p, gt))
            sUEO.append(computer_sUEO(gt, prediction.flatten(),output_main))
    metric_list = []
    ece_mean = sum(ece)/len(ece)
    sUEO_mean = sum(sUEO)/len(sUEO)

    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, gt == i))
    return metric_list,ece_mean,sUEO_mean

def test_single_volume_cct(image, label, net, classes, patch_size=[256, 256],n = 0, epoch = 0):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    ece = []
    sUEO = []
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                 0).unsqueeze(0).float().cuda()
            gt = label[ind,:,:].astype(int)
            gt = zoom(
                gt, (patch_size[0] / x, patch_size[1] / y), order=0)

            net.eval()
            with torch.no_grad():
                output_main = net(input)[0]
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                # 计算ECE
                p = torch.softmax(output_main, dim=1)
                p = transform(p).cpu().detach().numpy()
                ece.append(computer_ece(p, gt))
                sUEO.append(computer_sUEO(gt, out.flatten(),output_main))
                # print(,label))
                prediction[ind] = pred

    else:
        # input = torch.from_numpy(image).unsqueeze(
        #     0).unsqueeze(0).float().cuda()
        # x, y = image.shape[0], image.shape[1]
        # image = zoom(
        #     image, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        # gt = label.astype(int)
        # gt = zoom(
        #     gt, (patch_size[0] / x, patch_size[1] / y), order=0)
        net.eval()
        with torch.no_grad():
            output_main = net(input)[0]
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            # 计算ECE
            p = torch.softmax(output_main, dim=1)
            p = transform(p).cpu().detach().numpy()
            ece.append(computer_ece(p, gt))
            sUEO.append(computer_sUEO(gt, prediction.flatten(),output_main))
    metric_list = []
    ece_mean = sum(ece)/len(ece)
    sUEO_mean = sum(sUEO)/len(sUEO)

    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list,ece_mean,sUEO_mean

def add_gaussian_noise_np(array, mean=0, std=1):
    noise = np.random.randn(*array.shape) * std + mean
    return array.cpu() + noise

def test_single_volume_add_noise(image, label, net, classes, patch_size=[256, 256],n = 0, epoch = 0):
    # noise_m = torch.randn_like(image) * 0.1
    # image = torch.clamp(image + noise_m, 0, 1)
    # image = image + noise_m
#    noise_m = np.random.randn(*array.shape) * 1
#    image = image + noise_m
    image, label = image.squeeze(0).cpu().detach(
    ), label.squeeze(0).cpu().detach().numpy()
    ece = []
    sUEO = []
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            noise = np.random.randn(*slice.shape) * 0.15
            noisy_image = slice + noise


            # 保存为PNG图像
            slice_image = ((noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min()) * 255).astype(np.uint8)
            slice_image = Image.fromarray(slice_image)    
            png_folder_path = '/home/server/yyt/model/image'
            # 判断文件夹是否存在
            if not os.path.exists(png_folder_path):
                # 如果不存在则创建文件夹
                os.mkdir(png_folder_path)
            png_file_name = f'{n}_{ind}image.png'
            png_file_path = os.path.join(png_folder_path, png_file_name)
            slice_image.save(png_file_path)

            input = torch.from_numpy(noisy_image).unsqueeze(
                 0).unsqueeze(0).float().cuda()
            gt = label[ind,:,:].astype(int)
            gt = zoom(
                gt, (patch_size[0] / x, patch_size[1] / y), order=0)


            # yyt保存原始图象
            gt_image = ((gt - gt.min()) / (gt.max() - gt.min()) * 255).astype(np.uint8)
            gt_image = Image.fromarray(gt_image)     
            # 保存为PNG图像
            png_folder_path = '/home/server/yyt/model/gt'
            # 判断文件夹是否存在
            if not os.path.exists(png_folder_path):
                # 如果不存在则创建文件夹
                os.mkdir(png_folder_path)
            png_file_name = f'{n}_{ind}gt.png'
            png_file_path = os.path.join(png_folder_path, png_file_name)
            gt_image.save(png_file_path) 

            net.eval()
            with torch.no_grad():
                output_main = net(input)
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()

                # yyt生成mask
                pred = ((out - out.min()) / (out.max() - out.min()) * 255).astype(np.uint8)
                pred_image = Image.fromarray(pred)
                # 保存为PNG图像
                png_folder_path = '/home/server/yyt/model/result'
                # 判断文件夹是否存在
                if not os.path.exists(png_folder_path):
                    # 如果不存在则创建文件夹
                    os.mkdir(png_folder_path)
                png_file_name = f'{n}_{ind}pre_mask.png'
                png_file_path = os.path.join(png_folder_path, png_file_name)
                pred_image.save(png_file_path)

                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)

                # 计算ECE
                p = torch.softmax(output_main, dim=1)
                p = transform(p).cpu().detach().numpy()
                ece.append(computer_ece(p, gt))
                sUEO.append(computer_sUEO(gt, out.flatten(),output_main))
                # print(,label))
                prediction[ind] = pred

    else:
        # input = torch.from_numpy(image).unsqueeze(
        #     0).unsqueeze(0).float().cuda()
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float()
#        noise_m = np.random.randn(input) * 1
#    	   input = input + noise_m
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            # 计算ECE
            p = torch.softmax(output_main, dim=1)
            p = transform(p).cpu().detach().numpy()
            ece.append(computer_ece(p, gt))
            sUEO.append(computer_sUEO(gt, out,output_main))
    metric_list = []
    ece_mean = sum(ece)/len(ece)
    sUEO_mean = sum(sUEO)/len(sUEO)
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list,ece_mean,sUEO_mean

def test_single_volume_cct_add_noise(image, label, net, classes,noise, patch_size=[256, 256],n = 0, epoch = 0,):
    # noise_m = torch.randn_like(image) * 0.1
    # image = torch.clamp(image + noise_m, 0, 1)
    # image = image + noise_m
#    noise_m = np.random.randn(*array.shape) * 1
#    image = image + noise_m
    image, label = image.squeeze(0).cpu().detach(
    ), label.squeeze(0).cpu().detach().numpy()
    ece = []
    sUEO = []
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            noise = np.random.randn(*slice.shape) * noise
            noisy_image = slice + noise

            # 保存为PNG图像
            slice_image = ((noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min()) * 255).astype(np.uint8)
            slice_image = Image.fromarray(slice_image)    
            png_folder_path = '/home/server/yyt/model/image'
            # 判断文件夹是否存在
            if not os.path.exists(png_folder_path):
                # 如果不存在则创建文件夹
                os.mkdir(png_folder_path)
            png_file_name = f'{n}_{ind}image.png'
            png_file_path = os.path.join(png_folder_path, png_file_name)
            slice_image.save(png_file_path)

            input = torch.from_numpy(noisy_image).unsqueeze(
                 0).unsqueeze(0).float().cuda()
            gt = label[ind,:,:].astype(int)
            gt = zoom(
                gt, (patch_size[0] / x, patch_size[1] / y), order=0)

            # yyt保存原始图象
            gt_image = ((gt - gt.min()) / (gt.max() - gt.min()) * 255).astype(np.uint8)
            gt_image = Image.fromarray(gt_image)     
            # 保存为PNG图像
            png_folder_path = '/home/server/yyt/model/gt'
            # 判断文件夹是否存在
            if not os.path.exists(png_folder_path):
                # 如果不存在则创建文件夹
                os.mkdir(png_folder_path)
            png_file_name = f'{n}_{ind}gt.png'
            png_file_path = os.path.join(png_folder_path, png_file_name)
            gt_image.save(png_file_path) 

            net.eval()
            with torch.no_grad():
                output_main = net(input)[0]
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()

                # yyt生成mask
                pred = ((out - out.min()) / (out.max() - out.min()) * 255).astype(np.uint8)
                pred_image = Image.fromarray(pred)
                # 保存为PNG图像
                png_folder_path = '/home/server/yyt/model/result'
                # 判断文件夹是否存在
                if not os.path.exists(png_folder_path):
                    # 如果不存在则创建文件夹
                    os.mkdir(png_folder_path)
                png_file_name = f'{n}_{ind}pre_mask.png'
                png_file_path = os.path.join(png_folder_path, png_file_name)
                pred_image.save(png_file_path)

                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                # 计算ECE
                p = torch.softmax(output_main, dim=1)
                p = transform(p).cpu().detach().numpy()
                ece.append(computer_ece(p, gt))
                sUEO.append(computer_sUEO(gt, out.flatten(),output_main))
                # print(,label))
                prediction[ind] = pred

    else:
        # input = torch.from_numpy(image).unsqueeze(
        #     0).unsqueeze(0).float().cuda()
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float()
#        noise_m = np.random.randn(input) * 1
#    	   input = input + noise_m
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            # 计算ECE
            p = torch.softmax(output_main, dim=1)
            p = transform(p).cpu().detach().numpy()
            ece.append(computer_ece(p, gt))
            sUEO.append(computer_sUEO(gt, out,output_main))
    metric_list = []
    ece_mean = sum(ece)/len(ece)
    sUEO_mean = sum(sUEO)/len(sUEO)
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list,ece_mean,sUEO_mean

def test_single_volume_cct2(image, label, net, classes, patch_size=[256, 256],n = 0, epoch = 0):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]

            # yyt保存label
            label_slice = label[ind,:,:]
            label_slice = ((label_slice - label_slice.min()) / (label_slice.max() - label_slice.min()) * 255).astype(np.uint8)
            label_slice = Image.fromarray(label_slice)
            # 保存为PNG图像
            png_folder_path = 'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\gt'
            png_file_name = f'{n}_gt{ind}.png'
            png_file_path = os.path.join(png_folder_path, png_file_name)
            label_slice.save(png_file_path)

            # yyt保存原始图象
            slice_image = ((slice - slice.min()) / (slice.max() - slice.min()) * 255).astype(np.uint8)
            slice_image = Image.fromarray(slice_image)
            # 保存为PNG图像
            png_folder_path = 'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\image'
            png_file_name = f'{n}_image{ind}.png'
            png_file_path = os.path.join(png_folder_path, png_file_name)
            slice_image.save(png_file_path)

            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                 0).unsqueeze(0).float().cuda()

            net.eval()
            with torch.no_grad():
                output_main = net(input)[0]
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred


                # yyt生成mask
                pred = ((pred - pred.min()) / (pred.max() - pred.min()) * 255).astype(np.uint8)
                pred_image = Image.fromarray(pred)
                # 保存为PNG图像
                png_folder_path = f'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\result\\result_{epoch}'
                # 判断文件夹是否存在
                if not os.path.exists(png_folder_path):
                    # 如果不存在则创建文件夹
                    os.mkdir(png_folder_path)
                png_file_name = f'{n}_pre_mask{ind}.png'
                png_file_path = os.path.join(png_folder_path, png_file_name)
                pred_image.save(png_file_path)

                # yyt生成UN热力图
                # 计算预测掩码的不确定性熵
                uncertainty = Uentropy(output_main, 4)
                # 保存为PNG图像
                uncertainty = uncertainty.squeeze(0).cpu().detach().numpy()
                uncertainty = zoom(
                    uncertainty, (x / patch_size[0], y / patch_size[1]), order=0)
                # 将不确定性熵矩阵缩放到 [0, 255] 范围并保存为PNG图像
                uncertainty = ((uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()) * 255).astype(np.uint8)
                # 保存为PNG图像
                uncertainty_heatmap = cv2.applyColorMap(uncertainty.astype(np.uint8), cv2.COLORMAP_JET)
                UN_folder_path = f'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\UN_result\\UN_result_{epoch}'
                # 判断文件夹是否存在
                if not os.path.exists(UN_folder_path):
                    # 如果不存在则创建文件夹
                    os.mkdir(UN_folder_path)
                UN_file_name = f'{n}_UN_mask{ind}.png'
                UN_file_path = os.path.join(UN_folder_path, UN_file_name)
                cv2.imwrite(UN_file_path, uncertainty_heatmap)

    else:
        # input = torch.from_numpy(image).unsqueeze(
        #     0).unsqueeze(0).float().cuda()

        # yyt保存原始图象
        p_image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        p_image = Image.fromarray(p_image)
        # 保存为PNG图像
        png_folder_path = 'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\image'
        png_file_name = f'{n}_image.png'
        png_file_path = os.path.join(png_folder_path, png_file_name)
        p_image.save(png_file_path)

        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

            # yyt生成mask
            prediction = ((prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255).astype(np.uint8)
            pred_image = Image.fromarray(prediction)
            # 保存为PNG图像
            png_folder_path = f'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\result\\result_{epoch}'
            # 判断文件夹是否存在
            if not os.path.exists(png_folder_path):
                # 如果不存在则创建文件夹
                os.mkdir(png_folder_path)
            png_file_name = f'{n}_pre_mask.png'
            png_file_path = os.path.join(png_folder_path, png_file_name)
            pred_image.save(png_file_path)
            # yyt生成UN热力图
            # 计算预测掩码的不确定性熵
            uncertainty = Uentropy(output_main, 4)
            # 保存为PNG图像
            uncertainty = uncertainty.squeeze(0).cpu().detach().numpy()
            uncertainty = zoom(
                uncertainty, (x / patch_size[0], y / patch_size[1]), order=0)
            # 将不确定性熵矩阵缩放到 [0, 255] 范围并保存为PNG图像
            uncertainty = ((uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()) * 255).astype(np.uint8)
            # 保存为PNG图像
            uncertainty_heatmap = cv2.applyColorMap(uncertainty.astype(np.uint8), cv2.COLORMAP_JET)
            UN_folder_path = f'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\UN_result\\UN_result_{epoch}'
            # 判断文件夹是否存在
            if not os.path.exists(UN_folder_path):
                # 如果不存在则创建文件夹
                os.mkdir(UN_folder_path)
            UN_file_name = f'{n}_UN_mask.png'
            UN_file_path = os.path.join(UN_folder_path, UN_file_name)
            cv2.imwrite(UN_file_path, uncertainty_heatmap)

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_cct_test(image, label, net, classes, patch_size=[256, 256],n = 0):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]

            # yyt保存label
            label_slice = label[ind,:,:]
            label_slice = ((label_slice - label_slice.min()) / (label_slice.max() - label_slice.min()) * 255).astype(np.uint8)
            label_slice = Image.fromarray(label_slice)
            # 保存为PNG图像
            png_folder_path = 'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\gt'
            png_file_name = f'{n}_gt{ind}.png'
            png_file_path = os.path.join(png_folder_path, png_file_name)
            label_slice.save(png_file_path)

            # yyt保存原始图象
            slice_image = ((slice - slice.min()) / (slice.max() - slice.min()) * 255).astype(np.uint8)
            slice_image = Image.fromarray(slice_image)
            # 保存为PNG图像
            png_folder_path = 'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\image'
            png_file_name = f'{n}_image{ind}.png'
            png_file_path = os.path.join(png_folder_path, png_file_name)
            slice_image.save(png_file_path)

            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                 0).unsqueeze(0).float().cuda()

            net.eval()
            with torch.no_grad():
                output_main = net(input)[0]

                # 证据
                evidence_main = F.softplus(output_main)
                alpha_main = evidence_main + 1
                S_main = torch.sum(alpha_main, dim=1, keepdim=False)
                u = 4 / S_main

                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred


                # yyt生成mask
                pred = ((pred - pred.min()) / (pred.max() - pred.min()) * 255).astype(np.uint8)
                pred_image = Image.fromarray(pred)
                # 保存为PNG图像
                png_folder_path = f'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\result\\result'
                # 判断文件夹是否存在
                if not os.path.exists(png_folder_path):
                    # 如果不存在则创建文件夹
                    os.mkdir(png_folder_path)
                png_file_name = f'{n}_pre_mask{ind}.png'
                png_file_path = os.path.join(png_folder_path, png_file_name)
                pred_image.save(png_file_path)

                # yyt生成UN热力图
                # 计算预测掩码的不确定性熵
                uncertainty = Uentropy(output_main, 4)
                uncertainty = uncertainty.cpu().squeeze(0).detach().numpy()
                # 保存为PNG图像

                # 获取熵值大于0.5的
                for i in range(uncertainty.shape[0]):
                    for j in range(uncertainty.shape[1]):
                        if(uncertainty[i, j] > 0.5):
                            print("---------------")                   
                            print(alpha_main[:, 0,i, j])
                            print(alpha_main[:, 1,i, j])
                            print(alpha_main[:, 2,i, j])
                            print(alpha_main[:, 3,i, j])
                            print("---------------")
                            print(output_main[:, 0,i, j])
                            print(output_main[:, 1,i, j])
                            print(output_main[:, 2,i, j])
                            print(output_main[:, 3,i, j])
                            print("---------------")
                            print(u[:,i,j])
                            print("---------------")
                            print(torch.softmax(output_main, dim=1)[:, 0,i, j])
                            print(torch.softmax(output_main, dim=1)[:, 1,i, j])
                            print(torch.softmax(output_main, dim=1)[:, 2,i, j])
                            print(torch.softmax(output_main, dim=1)[:, 3,i, j])
                            print("---------------")
                            print(out[i,j])
                            print("---------------")


                uncertainty = zoom(
                    uncertainty, (x / patch_size[0], y / patch_size[1]), order=0)
                # 将不确定性熵矩阵缩放到 [0, 255] 范围并保存为PNG图像
                uncertainty = ((uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()) * 255).astype(np.uint8)
                # 保存为PNG图像
                uncertainty_heatmap = cv2.applyColorMap(uncertainty.astype(np.uint8), cv2.COLORMAP_JET)
                UN_folder_path = f'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\UN_result\\UN_result'
                # 判断文件夹是否存在
                if not os.path.exists(UN_folder_path):
                    # 如果不存在则创建文件夹
                    os.mkdir(UN_folder_path)
                UN_file_name = f'{n}_UN_mask{ind}.png'
                UN_file_path = os.path.join(UN_folder_path, UN_file_name)
                cv2.imwrite(UN_file_path, uncertainty_heatmap)

    else:
        # input = torch.from_numpy(image).unsqueeze(
        #     0).unsqueeze(0).float().cuda()

        # yyt保存原始图象
        p_image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        p_image = Image.fromarray(p_image)
        # 保存为PNG图像
        png_folder_path = 'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\image'
        png_file_name = f'{n}_image.png'
        png_file_path = os.path.join(png_folder_path, png_file_name)
        p_image.save(png_file_path)

        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

            # yyt生成mask
            prediction = ((prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255).astype(np.uint8)
            pred_image = Image.fromarray(prediction)
            # 保存为PNG图像
            png_folder_path = f'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\result\\result'
            # 判断文件夹是否存在
            if not os.path.exists(png_folder_path):
                # 如果不存在则创建文件夹
                os.mkdir(png_folder_path)
            png_file_name = f'{n}_pre_mask.png'
            png_file_path = os.path.join(png_folder_path, png_file_name)
            pred_image.save(png_file_path)
            # yyt生成UN热力图
            # 计算预测掩码的不确定性熵
            uncertainty = Uentropy(output_main, 4)
            # 保存为PNG图像
            uncertainty = uncertainty.squeeze(0).cpu().detach().numpy()
            uncertainty = zoom(
                uncertainty, (x / patch_size[0], y / patch_size[1]), order=0)
            # 将不确定性熵矩阵缩放到 [0, 255] 范围并保存为PNG图像
            uncertainty = ((uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()) * 255).astype(np.uint8)
            # 保存为PNG图像
            uncertainty_heatmap = cv2.applyColorMap(uncertainty.astype(np.uint8), cv2.COLORMAP_JET)
            UN_folder_path = f'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\UN_result\\UN_result'
            # 判断文件夹是否存在
            if not os.path.exists(UN_folder_path):
                # 如果不存在则创建文件夹
                os.mkdir(UN_folder_path)
            UN_file_name = f'{n}_UN_mask.png'
            UN_file_path = os.path.join(UN_folder_path, UN_file_name)
            cv2.imwrite(UN_file_path, uncertainty_heatmap)

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_cct_train(image, label, net, classes, patch_size=[256, 256],n = 0):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]

            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                 0).unsqueeze(0).float().cuda()

            # yyt保存原始图象
            p_image = ((slice - slice.min()) / (slice.max() - slice.min()) * 255).astype(np.uint8)
            p_image = Image.fromarray(p_image)
            # 保存为PNG图像
            png_folder_path = 'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\img'
            png_file_name = f'{n}_image.png'
            png_file_path = os.path.join(png_folder_path, png_file_name)
            p_image.save(png_file_path)

            net.eval()
            with torch.no_grad():
                output_main = net(input)[0]

                # 证据
                evidence_main = F.softplus(output_main)
                alpha_main = evidence_main + 1
                S_main = torch.sum(alpha_main, dim=1, keepdim=False)
                u = 4 / S_main

                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                
                # yyt生成mask
                prediction = ((prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255).astype(np.uint8)
                pred_image = Image.fromarray(prediction)
                # 保存为PNG图像
                png_folder_path = f'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\result\\res'
                # 判断文件夹是否存在
                if not os.path.exists(png_folder_path):
                    # 如果不存在则创建文件夹
                    os.mkdir(png_folder_path)
                png_file_name = f'{n}_pre_mask.png'
                png_file_path = os.path.join(png_folder_path, png_file_name)
                pred_image.save(png_file_path)
                # yyt生成UN热力图
                # 计算预测掩码的不确定性熵
                uncertainty = Uentropy(output_main, 4)
                # 保存为PNG图像
                uncertainty = uncertainty.squeeze(0).cpu().detach().numpy()
                uncertainty = zoom(
                    uncertainty, (x / patch_size[0], y / patch_size[1]), order=0)
                # 将不确定性熵矩阵缩放到 [0, 255] 范围并保存为PNG图像
                uncertainty = ((uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()) * 255).astype(np.uint8)
                # 保存为PNG图像
                uncertainty_heatmap = cv2.applyColorMap(uncertainty.astype(np.uint8), cv2.COLORMAP_JET)
                UN_folder_path = f'F:\\yyt\\WSL4MIS-main\\WSL4MIS-main\\data\\ACDC_our\\UN_result\\UN_res'
                # 判断文件夹是否存在
                if not os.path.exists(UN_folder_path):
                    # 如果不存在则创建文件夹
                    os.mkdir(UN_folder_path)
                UN_file_name = f'{n}_UN_mask.png'
                UN_file_path = os.path.join(UN_folder_path, UN_file_name)
                cv2.imwrite(UN_file_path, uncertainty_heatmap)


                # yyt生成UN热力图
                # 计算预测掩码的不确定性熵
                uncertainty = Uentropy(output_main, 4)
                uncertainty = uncertainty.cpu().squeeze(0).detach().numpy()
                # 保存为PNG图像

                # # 获取熵值大于0.5的
                # for i in range(label.shape[0]):
                #     for j in range(label.shape[1]):
                #         if((u[:,i,j] < 0.2) and (label[i,j] == 4)):
                #             print("---------------")                   
                #             print(alpha_main[:, 0,i, j])
                #             print(alpha_main[:, 1,i, j])
                #             print(alpha_main[:, 2,i, j])
                #             print(alpha_main[:, 3,i, j])
                #             print("---------------")
                #             # print(output_main[:, 0,i, j])
                #             # print(output_main[:, 1,i, j])
                #             # print(output_main[:, 2,i, j])
                #             # print(output_main[:, 3,i, j])
                #             # print("---------------")
                #             # print(u[:,i,j])
                #             # print("---------------")
                #             # print(torch.softmax(output_main, dim=1)[:, 0,i, j])
                #             # print(torch.softmax(output_main, dim=1)[:, 1,i, j])
                #             # print(torch.softmax(output_main, dim=1)[:, 2,i, j])
                #             # print(torch.softmax(output_main, dim=1)[:, 3,i, j])
                #             # print("---------------")
                #             # print(out[i,j])
                #             # print("---------------")

    else:
        # input = torch.from_numpy(image).unsqueeze(
        #     0).unsqueeze(0).float().cuda()

        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    return

