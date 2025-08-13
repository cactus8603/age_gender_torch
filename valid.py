import cv2
import numpy as np
import csv
import glob
from torchvision import transforms
import torch

import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os 
from model import TimmAgeGenderModel

import random
import math
import sys
import argparse
import logging
import re


AGE_STD = 8.61 # 25.86
AGE_MEAN = 25.86 # 8.61

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SquarePad:
    def __call__(self, img):
        w, h = img.size
        d = abs(h - w)
        pad = (d // 2, 0, d - d // 2, 0) if w < h else (0, d // 2, 0, d - d // 2)
        return transforms.functional.pad(img, pad, fill=0, padding_mode='edge')


def load_csv(filename):
    data = []
    with open(filename, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if '-1' not in row:
                data.append(row)
    return data

# 用你 val 的 transform（確保和訓練前處理一致）
val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    # SquarePad(),    
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

@torch.no_grad()
def predict_images(model, testfaces, model_path):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    
    model = model.load_checkpoint(model_path, device=device)
    model = model.to(device)
    model.eval()

    totalfacecount = 0
    correctgendercount = 0
    agemaearray = []

    results = []

    imagepaths = testfaces.keys()
    pbar = tqdm(total=len(imagepaths), initial=0)
    for imagepath in imagepaths:
        t_imagepath = './imags/' + os.path.basename(imagepath)
        srcimag = cv2.imread(t_imagepath)

        # p0 = next(model.parameters())
        # print('model param:', p0.dtype, p0.device)

        facedatas = testfaces[imagepath]
        for facedata in facedatas:
            fx = int(facedata[0])
            fy = int(facedata[1])
            fw = int(facedata[2])
            fh = int(facedata[3])
            gtgen = int(facedata[4])
            gtage = float(facedata[5])
            
            #print(fx, fy, fw, fh, gtgen, gtage)
            faceimag = srcimag[fy:fy+fh, fx:fx+fw,:].copy()

            faceimag = cv2.cvtColor(faceimag, cv2.COLOR_BGR2RGB)
            x = val_tf(faceimag).unsqueeze(0).to(device)

            # print(x.device)

            with autocast():
                g_logits, a_logits = model(x)
                # g_prob = torch.softmax(g_logits, dim=-1)[0].cpu().numpy()
                age   = a_logits.squeeze(-1).item()  # 換回歲數
                gender_idx = g_logits.argmax(dim=-1)  
            age = age * 8.61 + 25.86
            age *= 100

            # cv2.imwrite('faceimag.jpg', faceimag)
            
            # valid_input = [preprocess(img)]
            # _gender, _age = sess.run([y_conv1, sigmoid_age], feed_dict = {input_node : valid_input})
            # _gender = _gender[0][0] # batch, man
            # _age = _age[0][0] * 100 # age
            # print(gender_idx, gtgen)
            
            if gender_idx > 0 and gtgen > 0: # man
                correctgendercount += 1
            elif gender_idx == 0 and gtgen == 0: # woman
                correctgendercount += 1

            totalfacecount += 1
            agemae = abs(age - gtage)
            # print(age, gtage)
            agemaearray.append(agemae)

            results.append({
                "filename": imagepath,
                "bbox_x": fx,
                "bbox_y": fy,
                "bbox_w": fw,
                "bbox_h": fh,
                "gender": gender_idx,        # 0/1（依你的定義）
                "age": age
            })

            #print(_gender, gtgen, _age, gtage, agemae)
        pbar.update(1) # update

    with open("output.csv", mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    test_accuracy = correctgendercount / totalfacecount
    test_mae = np.mean(agemaearray)
    with open('accuracy.txt', 'a') as file:
        otxt = '%s %f %f\n' % (model_path, test_accuracy, test_mae)
        file.write(otxt)
    # print(test_accuracy, test_mae)

    print(f'acc:{test_accuracy*100}, mae:{test_mae}')


    # results = []
    # for p in (paths if isinstance(paths, (list,tuple)) else [paths]):
    #     img_bgr = cv2.imread(str(p))
    #     if img_bgr is None:
    #         results.append({"path": p, "error": "image not found"})
    #         continue
    #     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    #     x = val_tf(img_rgb).unsqueeze(0).to(device)

    #     with autocast():
    #         g_logits, a_logits = model(x)
    #         g_prob = torch.softmax(g_logits, dim=-1)[0].cpu().numpy()
    #         age   = a_logits.squeeze(-1).item() * age_scale  # 換回歲數

    #     gender_idx = int(g_prob.argmax())
    #     results.append({
    #         "path": p,
    #         "gender_index": gender_idx,        # 0/1（依你的定義）
    #         "gender_probs": g_prob.tolist(),   # [p_male, p_female] 或你的順序
    #         "age_years": age
    #     })
    # return results

if __name__ == '__main__':
    gtdata = load_csv('./gtdata/gtdata_combine.csv')

    model = TimmAgeGenderModel(model_name='mobilenetv3_large_100.ra_in1k') # mobilenetv3_small_100 efficientformerv2_s1
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    
    print("test set:", len(gtdata))
    testfaces = {}
    for gd in gtdata:
        filename = gd[0]
        if filename not in testfaces:
            testfaces[filename] = []
        testfaces[filename].append(gd[1:])

    # model_paths = glob.glob('./pretrain/*.pth')
    model_paths = glob.glob('./checkpoints/*.pth')
    model_paths.sort()
    # for n in range(len(models)):
    #     models[n] = os.path.splitext(models[n])[0]

    for model_path in model_paths:
        print(model_path)
        predict_images(model, testfaces, model_path)