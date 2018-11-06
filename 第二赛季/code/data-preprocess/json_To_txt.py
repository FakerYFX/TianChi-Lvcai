#coding=utf-8
import json
import cv2
import os

label_warp = {
              '不导电': "defect0",
              '擦花': "defect1",
              '角位漏底': "defect2",
              '桔皮': "defect3",
              '漏底': "defect4",
              '喷流': "defect5",
              '漆泡': "defect6",
              '起坑': "defect7",
              '杂色': "defect8",
              '脏点': "defect9",
              'normal':"noxiaci",
              }

def get_txtfiles(json_file_path):
    print(json_file_path)
    save_path = "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/data/guangdong_round2_train_20181011/noxiaci_txt/"
    jpeg_path = "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/data/guangdong_round2_train_20181011/noxiaci_jpg/"
    with open(json_file_path)as fr:
        for line in fr.readlines():
            #print(line)
            line = json.loads(line)
            #print(line)
            url = json_file_path.strip().split("/")
            img_name = url[-1].split(".")[0]
            height, width, _ = cv2.imread(jpeg_path + img_name + ".jpg").shape
            with open(save_path + img_name + ".txt", "w")as fw:
                for data in line["shapes"]:
                    temp_position = []
                    for bbox in data["points"]:
                        for xxx in bbox:
                            temp_position.append(xxx)
                    x1 = float(temp_position[0])
                    y1 = float(temp_position[1])
                    x2 = float(temp_position[2])
                    y2 = float(temp_position[3])
                    x3 = float(temp_position[4])
                    y3 = float(temp_position[5])
                    x4 = float(temp_position[6])
                    y4 = float(temp_position[7])
                    xx1 = min(x1, x2, x3, x4)
                    xx2 = max(x1, x2, x3, x4)
                    yy1 = min(y1, y2, y3, y4)
                    yy2 = max(y1, y2, y3, y4)
                    if xx2 > width:
                        xx2 = width
                    if yy2 > height:
                        yy2 = height
                    class_label = label_warp[data["label"].encode('utf-8')]
                    fw.write(img_name+" "+class_label+" "+str(xx1)+" "+str(xx2)+" "+str(yy1)+" "+str(yy2)+"\n")

def main():
    jsons_path = "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/data/guangdong_round2_train_20181011/noxiaci_json/"
    json_files = os.listdir(jsons_path)
    for json_file in json_files:
        get_txtfiles(jsons_path+json_file)

if __name__ == '__main__':
    main()