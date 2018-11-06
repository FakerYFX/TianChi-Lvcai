#coding=utf-8
import numpy as np
import datetime



model_list = ["../data/result/main_inception_v4.npy","../data/result/main_resnext_64.npy","../data/result/main_se_resnet50.npy","../data/result/main_senet154.npy"]
prob_list = [np.load(model) for model in model_list]
weight = [1.0] * len(model_list)

avg_res = np.zeros(prob_list[0].shape)
for prob in prob_list:
    avg_res += prob
avg_res = np.argmax(avg_res, axis=1)

res_last = avg_res.tolist()

img_path = []

label_warp = {0: "norm",
              1: "defect1",
              2: "defect2",
              3: "defect3",
              4: "defect4",
              5: "defect5",
              6: "defect6",
              7: "defect7",
              8: "defect8",
              9: "defect9",
              10: "defect10",
              11: "defect11",
              }

with open("../data/result/main_inception_v4/submission.csv/submission.csv")as fr:
    for line in fr.readlines():
        line = line.strip().split(',')
        img_path.append(line[0])
        
label_file = pd.DataFrame({'img_path': img_path, 'label': res_last})
label_file['label'] = label_file['label'].map(label_warp)

data.to_csv(("../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), header=None, index=False)

        
            
                
        



