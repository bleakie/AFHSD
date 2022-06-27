# -*- coding: utf-8 -*-
import cv2
import argparse
import os
import numpy as np
import json
import sys
import shutil
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils import load_label


def _train(args, subjects_train, interval_train, uid, fps=30, sample_fps=30):
    infos, annotations = [], []
    for ii in range(len(subjects_train)):
        npy_path = '%s/%s/%s/npy' % (args.root_path, args.dataset, args.mode)
        fileList = os.listdir(npy_path)
        img_file = None
        for file in fileList:
            if subjects_train[ii] in file:
                img_file = file
                break
        img_file = img_file.split('.')[0]
        frame_num = int(img_file.split('_')[-1])
        for jj in interval_train[ii]:
            start_frame, end_frame = jj[0], jj[1]
            infos.append((img_file, fps, sample_fps, frame_num, int(frame_num/(fps/sample_fps))))
            annotations.append((img_file, args.mode, 0, start_frame/fps, end_frame/fps, start_frame, end_frame))

    label_path = '%s/%s/%s/label' % (args.root_path, args.dataset, args.mode)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    df = pd.DataFrame(infos)
    df2 = pd.DataFrame(infos, columns=df.columns)
    save_path = '%s/%s_info.csv' % (label_path, uid)
    df2.to_csv(save_path, index=False)

    df = pd.DataFrame(annotations)
    df2 = pd.DataFrame(annotations, columns=df.columns)
    save_path = '%s/%s_annotations.csv' % (label_path, uid)
    df2.to_csv(save_path, index=False)


def _test(args, subjects_test, interval_test, uid, fps=30, sample_fps=30):
    data_info = {}
    for ii in range(len(subjects_test)):
        npy_path = '%s/%s/%s/npy' % (args.root_path, args.dataset, args.mode)
        fileList = os.listdir(npy_path)
        img_file = None
        for file in fileList:
            if subjects_test[ii] in file:
                img_file = file
                break
        img_file = img_file.split('.')[0]
        # if len(interval_test[ii]) < 1:
        data_info[img_file] = {"subset": "val", "annotations": [], 'sample_fps': fps,
                               'sample_count': int(img_file.split('_')[-1])}
        for jj in interval_test[ii]:
            data_info[img_file]["annotations"].append(
                {"segment": [jj[0]/fps, jj[1]/fps],
                 "label": args.mode,
                 "label_id": 0})
    label_path = '%s/%s/%s/label' % (args.root_path, args.dataset, args.mode)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    save_path = '%s/%s_val.json' % (label_path, uid)
    json_file = open(save_path, mode='w')
    json.dump(data_info, json_file, indent=4)

def gen_label(args):
    if args.dataset == 'CAS':
        fps, sample_fps = 30, 10
    elif args.dataset == 'SAMM':
        fps, sample_fps = 200, 200
    else:
        fps, sample_fps = 100, 100
    all_labels, all_paths, labels, paths, all_subjects = load_label(args.path_xlsx, args.dataset, args.mode)
    # train
    for one_subject in range(len(all_subjects)):
        subjects_train, interval_train = [], []
        subjects_test, interval_test = [], []
        for i in range(len(paths)):
            uid = paths[i].split('/')[-1].split('*')[0]
            if uid.split('_')[0] != all_subjects[one_subject]:
                subjects_train.append(uid)
                interval_train.append(labels[i])
            else:
                subjects_test.append(uid)
                interval_test.append(labels[i])
        _train(args, subjects_train, interval_train, all_subjects[one_subject], fps=fps, sample_fps=sample_fps)

    # for one_subject in range(len(all_subjects)):
    #     subjects_test, interval_test = [], []
    #     for i in range(len(all_paths)):
    #         uid = all_paths[i].split('/')[-1].split('*')[0]
    #         if uid.split('_')[0] == all_subjects[one_subject]:
    #             subjects_test.append(uid)
    #             if all_paths[i] in paths:
    #                 interval_test.append(all_labels[i])
    #             else:
    #                 interval_test.append([])
        # test
        _test(args, subjects_test, interval_test, all_subjects[one_subject], fps=fps, sample_fps=sample_fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, default="Macro",
                        help='Micro/Macro', metavar='N')
    parser.add_argument('--dataset', type=str, default="CAS",
                        help="SAMM/CAS/SMIC", metavar='N')
    parser.add_argument('--root_path', type=str,
                        default="/media/sai/data1/datasets/face/emotion/npy",
                        help="the path of CAS.xlsx/SAMM.xlsx", metavar='N')
    args = parser.parse_args()
    args.path_xlsx = '../../datasets/%s.xlsx' % (args.dataset)
    gen_label(args)
