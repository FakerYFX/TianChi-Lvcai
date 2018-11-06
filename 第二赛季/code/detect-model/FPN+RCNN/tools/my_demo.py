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
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import json
from utils.cython_bbox import bbox_overlaps

CLASSES = ('__background__',
           'defect0', 'defect1', 'defect2', 'defect3',
           'defect4', 'defect5', 'defect6', 'defect7', 'defect8',
           'defect9')


def bbox_vote(dets_NMS, dets_all, thresh=0.5):
    dets_voted = np.zeros_like(dets_NMS)   # Empty matrix with the same shape and type

    _overlaps = bbox_overlaps(
			np.ascontiguousarray(dets_NMS[:, 0:4], dtype=np.float),
			np.ascontiguousarray(dets_all[:, 0:4], dtype=np.float))

    # for each survived box
    for i, det in enumerate(dets_NMS):
        dets_overlapped = dets_all[np.where(_overlaps[i, :] >= thresh)[0]]
        assert(len(dets_overlapped) > 0)

        boxes = dets_overlapped[:, 0:4]
        scores = dets_overlapped[:, 4]

        out_box = np.dot(scores, boxes)

        dets_voted[i][0:4] = out_box / sum(scores)        # Weighted bounding boxes
        dets_voted[i][4] = det[4]                         # Keep the original score

        # Weighted scores (if enabled)
        if cfg.TEST.BBOX_VOTE_N_WEIGHTED_SCORE > 1:
            n_agreement = cfg.TEST.BBOX_VOTE_N_WEIGHTED_SCORE
            w_empty = cfg.TEST.BBOX_VOTE_WEIGHT_EMPTY

            n_detected = len(scores)

            if n_detected >= n_agreement:
                top_scores = -np.sort(-scores)[:n_agreement]
                new_score = np.average(top_scores)
            else:
                new_score = np.average(scores) * (n_detected * 1.0 + (n_agreement - n_detected) * w_empty) / n_agreement

            dets_voted[i][4] = min(new_score, dets_voted[i][4])

    return dets_voted

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    dict_small = dict()
    temp_list = []
    #print("inds.length"+str(len(inds)))
    if len(inds) == 0:
        return temp_list

    im = im[:, :, (2, 1, 0)]
    
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    h,w,_ = im.shape
   
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
    x1 = int(round(bbox[0])) if int(round(bbox[0])) > 0 else 0
    y1 = int(round(bbox[1])) if int(round(bbox[1])) > 0 else 0
    x2 = int(round(bbox[2])) if int(round(bbox[2])) < w else w
    y2 = int(round(bbox[3])) if int(round(bbox[3])) < h else h
    dict_small["xmin"] = x1
    dict_small["xmax"] = x2
    dict_small["ymin"] = y1
    dict_small["ymax"] = y2
    dict_small["confidence"]  = score.item()
    dict_small["label"] = class_name
    temp_list.append(dict_small)
    return temp_list
    #dict_temp["rects"].append(dict_small)

        

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join("../../data/guangdong_round2_test_b_20181106/", image_name)
    im = cv2.imread(im_file)
    h,w,_ = im.shape
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im,None,11)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    #CONF_THRESH = 0.05
    #NMS_THRESH = 0.3
    #print(image_name)
    #add by xyf
    dict_temp = dict()
    dict_temp["filename"] = im_name
    dict_temp["rects"] = list()
    
    flag = 0
    
    #get all boxes
    for j in xrange(1,11):
        #print(j)
        inds = np.where(scores[:, j] > 0.05)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j*4:(j+1)*4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(cls_dets, 0.3)
            
        dets_NMSed = cls_dets[keep, :]
        cls_dets = bbox_vote(dets_NMSed, cls_dets)
        #print(cls_dets.shape)
        #write answer
        index_new = np.where(cls_dets[:, -1] >= 0)[0]
        if len(index_new) == 0:
            #print("yes!!")
            continue
       
        my_temp_list = list()
        for kk in index_new:
            dict_small = dict()
            bbox = cls_dets[kk,:4]
            score = cls_dets[kk, -1]
            x1 = int(round(bbox[0])) if int(round(bbox[0])) > 0 else 0
            y1 = int(round(bbox[1])) if int(round(bbox[1])) > 0 else 0
            x2 = int(round(bbox[2])) if int(round(bbox[2])) < w else w
            y2 = int(round(bbox[3])) if int(round(bbox[3])) < h else h
            dict_small["xmin"] = x1
            dict_small["xmax"] = x2
            dict_small["ymin"] = y1
            dict_small["ymax"] = y2
            dict_small["confidence"]  = score.item()
            dict_small["label"] = CLASSES[j]
            temp_str = str(x1)+" "+str(x2)+" "+str(y1)+" "+str(y2)+" "+str(score.item())
            #print(temp_str)
            if temp_str in my_temp_list:
                #print("has been in!")
                continue
            else:      
                my_temp_list.append(temp_str)
                dict_temp["rects"].append(dict_small)
        
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
#     parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
#                         choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/models/pascal_voc/FPN/FP_Net_end2end/test.prototxt"
    
    caffemodel = "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/output/shared_rcnn_anchor_aug_all/voc_2007_trainval/fpn_iter_70000.caffemodel"

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        #caffe.set_device(args.gpu_id)
        #cfg.GPU_ID = args.gpu_id
        caffe.set_device(0)
        cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
#     im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
#     for i in xrange(2):
#         _, _= im_detect(net, im)

    im_names = os.listdir('../../data/guangdong_round2_test_b_20181106')
    print(len(im_names))
    dict_save = dict()
    dict_save["results"] = list()
    num_zero = 0
    for im_name in im_names:
        print(im_name)
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        dict_temp,flag = demo(net, im_name)
        dict_save["results"].append(dict_temp)
        if flag == 1:
            num_zero = num_zero+1
    with open('7W_last1.json', 'w') as fw:
        json.dump(dict_save,fw)
    print num_zero
