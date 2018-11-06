import os
import math
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm

# test data
test_img_path = []
label = []
with open("human_9319.csv")as fr:
    for line in fr.readlines():
        line = line.strip().split(",")
        test_img_path = "data/guangdong_round1_test_a_20180916/"+line[0]
        if line[1]=='norm':
            label.append(0)
        elif line[1]=='defect1':
            label.append(1)
        elif line[1]=='defect2':
            label.append(2)
        elif line[1]=='defect3':
            label.append(3)
        elif line[1]=='defect4':
            label.append(4)
        elif line[1]=='defect5':
            label.append(5)
        elif line[1]=='defect6':
            label.append(6)
        elif line[1]=='defect7':
            label.append(7)
        elif line[1]=='defect8':
            label.append(8)
        elif line[1]=='defect9':
            label.append(9)
        elif line[1]=='defect10':
            label.append(10)
        elif line[1]=='defect11':
            label.append(11)
                        
test_file = pd.DataFrame({'img_path': test_img_path,'label': label})

test_file.to_csv('data/label_train3.csv', index=False)