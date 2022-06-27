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
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, default="Micro",
                        help='Micro/Macro', metavar='N')
    parser.add_argument('--dataset', type=str, default="SAMM",
                        help="SAMM/CAS/SMIC", metavar='N')
    args = parser.parse_args()
    path_xlsx = '../../datasets/%s.xlsx' % (args.dataset)
    if args.dataset == 'CAS':
        label, path = read_xlsx_cas(path_xlsx, args.mode)
    elif args.dataset == 'SAMM':
        label, path = read_xlsx_samm(path_xlsx, args.mode)
    else:
        label, path = read_xlsx_smic(path_xlsx, args.mode)
    K = cal_k(label)
