import os
import multiprocessing as mp
import argparse
import cv2
import numpy as np



def sub_processor(args):

    files = sorted(os.listdir(args.img_dir))
    for file in files:
        npy_path = os.path.join(args.output_dir, 'npy')
        if not os.path.exists(npy_path):
            os.makedirs(npy_path)
        target_file = '%s/%s.npy' % (npy_path, file)
        imgs = []
        imgList = os.listdir(os.path.join(args.img_dir, file))
        imgList.sort()
        for im in imgList:
            img = cv2.imread(os.path.join(args.img_dir, file, im))
            img = cv2.resize(img, (args.resize[1], args.resize[0]))
            imgs.append(img[:, :, ::-1])
        imgs = np.stack(imgs)
        if imgs.shape[0] > args.max_frame_num:
            imgs = imgs[:args.max_frame_num]
        np.save(target_file, imgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='/media/sai/data01/emotion/micro_macro_datasets/SPOT/CAS_Micro/')
    parser.add_argument('--output_dir', type=str, default='/media/sai/data01/emotion/micro_macro_datasets/SPOT/npy/CAS/Micro')
    parser.add_argument('--resize', type=int, default=[96, 96])
    parser.add_argument('--max_frame_num', type=int, default=10000)
    args = parser.parse_args()

    sub_processor(args)
