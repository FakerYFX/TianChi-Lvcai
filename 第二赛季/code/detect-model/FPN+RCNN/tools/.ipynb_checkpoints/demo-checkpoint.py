#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import json
# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

# NETS = {'vgg16': ('VGG16',
#                   'VGG16_faster_rcnn_final.caffemodel'),
#         'zf': ('ZF',
#                   'ZF_faster_rcnn_final.caffemodel')}

class_name = {}
class_name[0]="background"
class_name[1]="defect0"
class_name[2]="defect1"
class_name[3]="defect2"
class_name[4]="defect3"
class_name[5]="defect4"
class_name[6]="defect5"
class_name[7]="defect6"
class_name[8]="defect7"
class_name[9]="defect8"
class_name[10]="defect9"

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

#         ax.add_patch(
#             plt.Rectangle((bbox[0], bbox[1]),
#                           bbox[2] - bbox[0],
#                           bbox[3] - bbox[1], fill=False,
#                           edgecolor='red', linewidth=3.5)
#             )
#         ax.text(bbox[0], bbox[1] - 2,
#                 '{:s} {:.3f}'.format(class_name, score),
#                 bbox=dict(facecolor='blue', alpha=0.5),
#                 fontsize=14, color='white')

#     ax.set_title(('{} detections with '
#                   'p({} | box) >= {:.1f}').format(class_name, class_name,
#                                                   thresh),
#                   fontsize=14)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    thresh=0.8
    # Load the demo image
    im_file = os.path.join("../../data/guangdong_round2_test_a_20181011/", image_name)
    im = cv2.imread(im_file)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im,None,11)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    #add some thing
    dict_temp = dict()
    dict_temp["filename"] = im_name
    dict_temp["rects"] = list()
    imj =im
    h,w,_ = im.shape
    dict_small = dict()
    flag = 0
    for jj in xrange(1, 11):  
        #print(str(jj))
        indsj = np.where(scores[:, jj] > thresh)[0]
        cls_scoresj = scores[indsj, jj]
        cls_boxesj = boxes[indsj, jj*4:(jj+1)*4]
        cls_detsj = np.hstack((cls_boxesj, cls_scoresj[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
        keep = nms(cls_detsj, cfg.TEST.NMS)
        cls_detsj = cls_detsj[keep, :]
        detsj = cls_detsj
        for ii in xrange(np.minimum(10, detsj.shape[0])):
            bboxj = detsj[ii, :4]
            scorej = detsj[ii, -1]
            if scorej > 0.1:
                if bboxj != []:
                    x1 = bboxj[0]
                    y1 = bboxj[3]
                    x2 = bboxj[2]
                    y2 = bboxj[1]
                    x1 = int(round(x1)) if int(round(x1)) > 0 else 0
                    y1 = int(round(y1)) if int(round(y1)) > 0 else 0
                    x2 = int(round(x2)) if int(round(x2)) < w else w
                    y2 = int(round(y2)) if int(round(y2)) < h else h
                    dict_small["xmin"] = x1
                    dict_small["xmax"] = x2
                    dict_small["ymin"] = y1
                    dict_small["ymax"] = y2
                    dict_small["confidence"]  = scorej.item()
                    dict_small["label"] = class_name[jj]
                    dict_temp["rects"].append(dict_small)
            else:
                continue
    if len(dict_temp["rects"])==0:
        flag = 1
    return dict_temp,flag
                
                    
                    

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    #parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        #choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/models/pascal_voc/FPN/FP_Net_end2end/test.prototxt"
    
    caffemodel = "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/output/FP_Net_end2end/voc_2007_trainval/fpn_iter_30000.caffemodel"

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
#     im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
#     for i in xrange(2):
#         _, _= im_detect(net, im)

#     im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
#                 '001763.jpg', '004545.jpg']

    im_names = os.listdir('../../data/guangdong_round2_test_a_20181011')
    print(len(im_names))
    dict_save = dict()
    dict_save["results"] = list()
    num_zero = 0
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        dict_temp,flag = demo(net, im_name)
        dict_save["results"].append(dict_temp)
        if flag == 1:
            num_zero = num_zero+1
    
    with open('fpn_0.1.json', 'w') as fw:
        json.dump(dict_save,fw)
    print num_zero
