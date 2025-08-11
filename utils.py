import numpy as np
import cv2
import csv
import random
import math
import glob
import sys
import argparse
import logging
import re
import os
from tqdm import tqdm
import pandas as pd


def getparentid(spcatid):
    if spcatid < 9: # no mask man
        return 0
    else: # no mask female
        return 1

def get_training_data():

    # クラス数
    parent_class_num = 2
    class_num = 18

    categorycorr = {}
    with open('./dataset/category.txt') as fp:
        lines = fp.readlines()
        for n in range(1, len(lines)):
            line = lines[n].rstrip('\n')
            word = line.split(' ')
            cid = int(word[0])
            word.pop(0)
            categorycorr[cid] = word
    for key in categorycorr:
        print(key, categorycorr[key])

    os.makedirs("./logs/", exist_ok=True)

    alldata = []
    alldata_tmp = []

    csv_paths = [
        "dataset/afad_labels.csv",
        "dataset/appa_labels.csv",
        "dataset/fairface_train_labels.csv",
        "dataset/fairface_val_labels.csv",
        "dataset/utk_labels.csv",

    ]

    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # df["file"] = df["file"].astype(str)
    df = df.to_numpy().tolist()

    for idx, row in tqdm(enumerate(df), total=len(df)):
        file_path, age, gender, race = row

        catid = int(age)
        iage = int(age)

        age = float(iage / 100.0)
        catid = int(iage // 10)

        if gender == 0: catid += 9
        if gender == 1 and catid > 8: # male
            continue
        elif gender == 0 and (catid < 9 or catid > 17):
            continue 

        if catid > 17: continue

        spcatid = int(categorycorr[catid][0])
        catname = categorycorr[catid][1]
        spcatname = categorycorr[catid][2]
        label = [0]*class_num
        label[spcatid] = 1
        ptlabel = [0]*parent_class_num
        ptid = getparentid(spcatid)
        ptlabel[ptid] = 1

        alldata_tmp.append([file_path, label, ptlabel, catid, spcatid, catname, spcatname, age])
        alldata.append([file_path, gender, age])

    totalcount = len(alldata)
    print('++++++++++++++++++++ total', totalcount)

    random.shuffle(alldata)


    spcatcnt = [0]*class_num
    for d in alldata_tmp:
        spcatcnt[d[4]] += 1
    for i, spc in enumerate(spcatcnt):
        print(i, spc)

    # alldata = alldata.to_numpy().tolist()
    traincount = int(totalcount * 0.8)
    # validcount = int((totalcount - traincount) * 0.1)

    data_set = alldata[:traincount]
    test_set = alldata[traincount:]
    # test_set = alldata[validcount:]
    
    print("data set:", len(data_set))
    print("test set:", len(test_set))

    return data_set, test_set
