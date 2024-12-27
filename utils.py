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

def getparentid(spcatid):
    if spcatid < 9: # no mask man
        return 0
    else: # no mask female
        return 1

def load_csv(filename):
    data = []
    with open(filename, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # 讀取標題
        # data.append(header)  # 如果你想保留標題行
        for row in csv_reader:
            if '-1' not in row and float(row[1]) < 90:  # 檢查該列是否有 -1
                data.append(row)  # 將每一行加入列表
    return data

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

    datapath = glob.glob('./dataset/AAFD/aglined_faces/*.jpg')
    for dpath in datapath:
        # print((type(dpath)))
        # match = re.search(r'.+dataset\/AAFD\/aglined_faces\/(\d+)\/.+\.jpg', dpath)
        # match = re.search(r'.+dataset\/AAFD\/aglined_faces\/(\d+)A(\d+)\.jpg', dpath)
        match = re.search(r'.+dataset[\\\\/]AAFD[\\\\/]aglined_faces[\\\\/](\d+)A(\d+)\.jpg', dpath)
        catid = int(match.group(1))
        iage = int(match.group(2))
        # print(catid, iage)
 
        age = iage / 100.0
        catid = int(iage // 10)

        # female: 0, male: 1
        if int(match.group(1)) >= 7381:
            gender = 1
        else: 
            gender = 0
            catid += 9

        # check gender and category is correct
        if gender == 1 and catid > 8: # male
            continue
        elif gender == 0 and (catid < 9 or catid > 17):
            continue 

        # if catid > 17:
        #     continue
        spcatid = int(categorycorr[catid][0])
        catname = categorycorr[catid][1]
        spcatname = categorycorr[catid][2]
        label = [0]*class_num
        label[spcatid] = 1
        ptlabel = [0]*parent_class_num
        ptid = getparentid(spcatid)
        ptlabel[ptid] = 1
        # match = re.search(r'.+dataset\/AAFD\/aglined_faces\/\d+\/.....A(\d+)_\d+\.jpg', dpath)
        age = float(match.group(2)) / 100

        # print(dpath)
        # print(gender, ptlabel)
        # alldata.append([dpath, label, ptlabel, catid, spcatid, catname, spcatname, age])
        alldata.append([dpath, gender, age])

    totalcount = len(alldata)
    print('++++++++++++++++++++ total', totalcount)

    datapath = glob.glob('./dataset/AFAD-Full/*/*/*.jpg')
    for dpath in datapath:
        # print(dpath)
        # match = re.search(r'.+dataset/AFAD-Full/(\d+)/(\d+)/.+\.jpg', d)
        match = re.search(r'.+dataset[\\\\/]AFAD-Full[\\\\/](\d+)[\\\\/](\d+)[\\\\/].+\.jpg', dpath)
        iage = int(match.group(1))
        igender = int(match.group(2))
        age = iage / 100.0
        catid = int(iage // 10)

        # 111: male, 112: female
        if igender == 111:
            gender = 1
        else:
            gender = 0
            catid += 9
        
        # check gender and category is correct
        if gender == 1 and catid > 8: # male
            continue
        elif gender == 0 and (catid < 9 or catid > 17):
            continue 

        spcatid = int(categorycorr[catid][0])
        catname = categorycorr[catid][1]
        spcatname = categorycorr[catid][2]
        label = [0]*class_num
        label[spcatid] = 1
        ptlabel = [0]*parent_class_num
        ptid = getparentid(spcatid)
        ptlabel[ptid] = 1
        # alldata.append([dpath, label, ptlabel, catid, spcatid, catname, spcatname, age])
        alldata.append([dpath, gender, age])

    random.shuffle(alldata)

    totalcount = len(alldata)
    print('++++++++++++++++++++ total', totalcount)

    # # ### add utk and load annotations of utk dataset ###
    # anno_path = './dataset/utk/annotation/utk_face_train.csv'
    # utk_train = load_csv(anno_path)
    # for d in utk_train:
    #     # print(os.path.basename(d[0]))
    #     dpath = os.path.join('./dataset/utk/images', d[0])

    #     if not os.path.exists(dpath):
    #         # print(dpath)
    #         continue
        
    #     # select asian 
    #     # if os.path.basename(d[0]).split('_')[2] == '2':
    #         # print(os.path.basename(d[0]))
    #     # print('\n')

    #     # print(match.group(0))
    #     # print(match.group(1))
    #     # print(match.group(2))

    #     # print(type(match.group(2)))
    #     # for itm in match:
    #     #     print(itm)
    #     # asian = match.group(2)
    #     # print(asian)

    #     iage = int(d[1])
    #     age = iage / 100.0
    #     catid = int(iage // 10)
    #     # print(catid)
    #     if d[2] == 'M':
    #         gender = 1
    #     else: 
    #         gender = 0
    #         catid += 9

    #     # check gender and category is correct
    #     if gender == 1 and catid > 8: # male
    #         continue
    #     elif gender == 0 and (catid < 9 or catid > 17):
    #         continue 

    #     spcatid = int(categorycorr[catid][0])
    #     catname = categorycorr[catid][1]
    #     spcatname = categorycorr[catid][2]
    #     label = [0]*class_num
    #     label[spcatid] = 1
    #     ptlabel = [0]*parent_class_num
    #     ptid = getparentid(spcatid)
    #     ptlabel[ptid] = 1
    #     # print(ptlabel)
    #     alldata.append([dpath, label, ptlabel, catid, spcatid, catname, spcatname, age])
    #     # print(dpath)
    #     # print(gender, ptlabel)
    # random.shuffle(alldata)
    # # ### add utk and load annotations of utk dataset ###

    # totalcount = len(alldata)
    # print('++++++++++++++++++++ total add asian', totalcount)

    # datapath = glob.glob('./dataset/MORPH_2')
    # train_anno_path = './dataset/MORPH_2/Index/Train.csv'
    # morph_train = load_csv(train_anno_path)
    # for d in morph_train:
    #     dpath = os.path.relpath(d[3], "/kaggle/input/morph/Dataset")
    #     dpath = os.path.join('./dataset/MORPH_2', dpath)
    #     # print(dpath)
    #     if not os.path.exists(dpath):
    #         # print(dpath)
    #         continue
    #     # print(dpath)
        
    #     iage = int(d[0])
    #     age = iage / 100
    #     catid = int(iage // 10)
    #     # gender = d[1]

    #     if d[1] == '0': 
    #         gender = 0
    #         catid += 9
    #     elif d[1] == '1': 
    #         gender = 1
        
    #     # check gender and category is correct
    #     if gender == 1 and catid > 8: # male
    #         continue
    #     elif gender == 0 and (catid <  or catid > 17):
    #         continue 
            
            
    #     spcatid = int(categorycorr[catid][0])
    #     catname = categorycorr[catid][1]
    #     spcatname = categorycorr[catid][2]
    #     label = [0]*class_num
    #     label[spcatid] = 1
    #     ptlabel = [0]*parent_class_num
    #     ptid = getparentid(spcatid)
    #     ptlabel[ptid] = 1
    #     # print(dpath)
    #     # print(gender, ptlabel)
    #     alldata.append([dpath, label, ptlabel, catid, spcatid, catname, spcatname, age])
    # random.shuffle(alldata)

    # totalcount = len(alldata)
    # print('++++++++++++++++++++ total MORPH', totalcount)

    # fairface
    train_anno_path = './dataset/fairface/reduced_data_train_20.csv'
    train_path = './dataset/fairface/'
    fair_train = load_csv(train_anno_path)

    for d in fair_train:
        dpath = os.path.join(train_path, d[0])
        if not os.path.exists(dpath):
            # print(dpath)
            continue

        iage = float(d[1])
        age = iage / 100
        catid = int(iage // 10)

        if d[2] == 'Female': 
            gender = 0
            catid += 9
        elif d[2] == 'Male': 
            gender = 1

        # check gender and category is correct
        if gender == 1 and catid > 8: # male
            continue
        elif gender == 0 and (catid < 9 or catid > 17):
            continue 

        spcatid = int(categorycorr[catid][0])
        catname = categorycorr[catid][1]
        spcatname = categorycorr[catid][2]
        label = [0]*class_num
        label[spcatid] = 1
        ptlabel = [0]*parent_class_num
        ptid = getparentid(spcatid)
        ptlabel[ptid] = 1
        # print(dpath)
        # print(gender, ptlabel)
        # alldata.append([dpath, label, ptlabel, catid, spcatid, catname, spcatname, age])
        alldata.append([dpath, gender, age])
    random.shuffle(alldata)

    totalcount = len(alldata)
    print('++++++++++++++++++++', totalcount)

    val_anno_path = './dataset/fairface/reduced_data_val_20.csv'
    val_path = './dataset/fairface/'
    fair_val = load_csv(val_anno_path)

    for d in fair_val:
        dpath = os.path.join(val_path, d[0])
        if not os.path.exists(dpath):
            # print(dpath)
            continue

        iage = float(d[1])
        age = iage / 100
        catid = int(iage // 10)

        if d[2] == 'Female': 
            gender = 0
            catid += 9
        elif d[2] == 'Male': 
            gender = 1

        # check gender and category is correct
        if gender == 1 and catid > 8: # male
            continue
        elif gender == 0 and (catid < 9 or catid > 17):
            continue 

        spcatid = int(categorycorr[catid][0])
        catname = categorycorr[catid][1]
        spcatname = categorycorr[catid][2]
        label = [0]*class_num
        label[spcatid] = 1
        ptlabel = [0]*parent_class_num
        ptid = getparentid(spcatid)
        ptlabel[ptid] = 1
        # print(dpath)
        # print(gender, ptlabel)
        # alldata.append([dpath, label, ptlabel, catid, spcatid, catname, spcatname, age])
        alldata.append([dpath, gender, age])
    random.shuffle(alldata)

    totalcount = len(alldata)
    print('++++++++++++++++++++', totalcount)

    anno_path = './dataset/adience/annotations/processed_adience.csv'
    path = './dataset/adience/faces'
    adience_train = load_csv(anno_path)

    for d in adience_train:
        dpath = os.path.join(path, d[0])
        # print(dpath)
        # print(os.path.exists(dpath))
        if not os.path.exists(dpath):
            # print(dpath)
            continue

        iage = float(d[1])
        age = iage / 100
        catid = int(iage // 10)

        if d[2] == 'F': 
            gender = 0
            catid += 9
        elif d[2] == 'M': 
            gender = 1

        # check gender and category is correct
        if gender == 1 and catid > 8: # male
            continue
        elif gender == 0 and (catid < 9 or catid > 17):
            continue 

        spcatid = int(categorycorr[catid][0])
        catname = categorycorr[catid][1]
        spcatname = categorycorr[catid][2]
        label = [0]*class_num
        label[spcatid] = 1
        ptlabel = [0]*parent_class_num
        ptid = getparentid(spcatid)
        ptlabel[ptid] = 1
        # print(dpath)
        # print(gender, ptlabel)
        # alldata.append([dpath, label, ptlabel, catid, spcatid, catname, spcatname, age])
        alldata.append([dpath, gender, age])
    random.shuffle(alldata)

    totalcount = len(alldata)
    print('++++++++++++++++++++', totalcount)


    # spcatcnt = [0]*class_num
    # for d in alldata:
    #     spcatcnt[d[4]] += 1
    # for i, spc in enumerate(spcatcnt):
    #     print(i, spc)

    traincount = int(totalcount * 0.9)
    validcount = totalcount - traincount

    data_set = alldata[:traincount]
    test_set = alldata[traincount:]
    
    print("data set:", len(data_set))
    print("test set:", len(test_set))

    # fp = open('testlist.txt', 'w')
    # for ts in test_set:
    #     ftxt = '%s,%d,%d,%s,%s\n' % (ts[0], ts[3], ts[4], ts[5], ts[6])
    #     fp.write(ftxt)
    # fp.close()

    return data_set, test_set
