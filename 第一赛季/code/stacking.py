# -*- coding: utf-8 -*-
'''
Created on Thu Sep 20 16:16:39 2018
 
@ author: herbert-chen
'''
import os
import time
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import model_v4
import resnext
import senet


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    # 随机种子
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    random.seed(666)
    # 默认使用PIL读图
    def default_loader(path):
        # return Image.open(path)
        return Image.open(path).convert('RGB')
    
    # 验证集图片读取
    class ValDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, label = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)

    # 测试集图片读取
    class TestDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, filename

        def __len__(self):
            return len(self.imgs)

    # 数据增强：在给定角度中随机进行旋转
    class FixedRotation(object):
        def __init__(self, angles):
            self.angles = angles

        def __call__(self, img):
            return fixed_rotate(img, self.angles)

    def fixed_rotate(img, angles):
        angles = list(angles)
        angles_num = len(angles)
        index = random.randint(0, angles_num - 1)
        return img.rotate(angles[index])
    
    # 验证函数
    def validate(val_loader, model, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (images, labels) in enumerate(val_loader):
            image_var = torch.tensor(images).cuda(async=True)
            target = torch.tensor(labels).cuda(async=True)

            # 图片前传。验证和测试时不需要更新网络权重，所以使用torch.no_grad()，表示不计算梯度
            with torch.no_grad():
                y_pred = model(image_var)
                loss = criterion(y_pred, target)

            # measure accuracy and record loss
            prec, PRED_COUNT = accuracy(y_pred.data, labels, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            acc.update(prec, PRED_COUNT)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('TrainVal: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))

        print(' * Accuray {acc.avg:.3f}'.format(acc=acc), '(Previous Best Acc: %.3f)' % best_precision,
              ' * Loss {loss.avg:.3f}'.format(loss=losses), 'Previous Lowest Loss: %.3f)' % lowest_loss)
        return acc.avg, losses.avg

    # 测试函数
    def test(main_inception_v4,test_loader, model):
        csv_map = OrderedDict({'filename': [], 'probability': []})
        # switch to evaluate mode
        model.eval()
        for i, (images, filepath) in enumerate(tqdm(test_loader)):
            # bs, ncrops, c, h, w = images.size()
            filepath = [os.path.basename(i) for i in filepath]
            image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4

            with torch.no_grad():
                y_pred = model(image_var)
                if i==0:
                    result_last = y_pred
                else:
                    result_last = torch.cat((result_last,y_pred), 0)
                # 使用softmax函数将图片预测结果转换成类别概率
                smax = nn.Softmax(1)
                smax_out = smax(y_pred)
                
            # 保存图片名称与预测概率
            csv_map['filename'].extend(filepath)
            for output in smax_out:
                prob = ';'.join([str(i) for i in output.data.tolist()])
                csv_map['probability'].append(prob)
                
        result_path = '../data/result/' + '{}'.format(main_inception_v4)
        np.save('{}.npy'.format(result_path), result_last)         
        #print(result_last.shape)
        result = pd.DataFrame(csv_map)
        result['probability'] = result['probability'].map(lambda x: [float(i) for i in x.split(';')])

        # 转换成提交样例中的格式
        sub_filename, sub_label = [], []
        for index, row in result.iterrows():
            sub_filename.append(row['filename'])
            pred_label = np.argmax(row['probability'])
            if pred_label == 0:
                sub_label.append('norm')
            else:
                sub_label.append('defect%d' % pred_label)

        # 生成结果文件，保存在result文件夹中，可用于直接提交
#         submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
#         submission.to_csv('./result/%s/submission.csv' % main_inception_v4, header=None, index=False)
        return 


    # 用于计算精度和时间的变化
    class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
          
    # 设定GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    batch_size = 8   
    # 进程数量，最好不要超过电脑最大进程数，尽量能被batch size整除。windows下报错可以改为workers=0
    workers = 8
    
    # 图片归一化，由于采用ImageNet预训练网络，因此这里直接采用ImageNet网络的参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 读取测试图片列表
    test_data_list = pd.read_csv('../data/test.csv')
    # 测试集图片变换
    test_data = TestDataset(test_data_list,
                            transform=transforms.Compose([
                                transforms.Resize((400, 400)),
                                transforms.CenterCrop(384),
                                transforms.ToTensor(),
                                normalize,
                            ]))
    test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)
    # 创建inception_v4模型
    model = model_v4.v4(num_classes=12)
    model = torch.nn.DataParallel(model).cuda()
    
    # 读取最佳模型，预测测试集，并生成可直接提交的结果文件
    model_name = "../data/model/main_iv4_enhance/model_best.pth.tar"
    main_inception_v4 = "main_iv4_enhance"
    best_model = torch.load(model_name)
    model.load_state_dict(best_model['state_dict'])
    test(main_inception_v4,test_loader=test_loader, model=model)

    # 创建senet154模型
    model = senet.senet_new_154(num_classes=12)
    model = torch.nn.DataParallel(model).cuda()
    
    # 读取最佳模型，预测测试集，并生成可直接提交的结果文件
    model_name = "../data/model/main_senet154_enhance/model_best.pth.tar"
    main_inception_v4 = "main_senet154_enhance"
    best_model = torch.load(model_name)
    model.load_state_dict(best_model['state_dict'])
    test(main_inception_v4,test_loader=test_loader, model=model)

    # 创建resnext64模型
    model = resnext.resnex64(num_classes=12)
    model = torch.nn.DataParallel(model).cuda()
    
    # 读取最佳模型，预测测试集，并生成可直接提交的结果文件
    model_name = "../data/model/main_resnext_64_enhance/model_best.pth.tar"
    main_inception_v4 = "main_resnext_64_enhance"
    best_model = torch.load(model_name)
    model.load_state_dict(best_model['state_dict'])
    test(main_inception_v4,test_loader=test_loader, model=model)

    # 创建se_resnet_50模型
    model = senet.my_se_resnet50(num_classes=12)
    model = torch.nn.DataParallel(model).cuda()
    
    # 读取最佳模型，预测测试集，并生成可直接提交的结果文件
    model_name = "../data/model/main_se_resnet50/model_best.pth.tar"
    main_inception_v4 = "main_se_resnet50"
    best_model = torch.load(model_name)
    model.load_state_dict(best_model['state_dict'])
    test(main_inception_v4,test_loader=test_loader, model=model)
    

    # 释放GPU缓存
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
