import torch
import torch.nn as nn
import os
import numpy as np
import argparse
import json
import sys
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from AFSD.common import videotransforms
from AFSD.BDNet import BDNet
from AFSD.common.find_result import softnms_v2
from AFSD.common.config import config
from AFSD.common.calculate import cal_IOU
from AFSD.common.find_result import recursive_merge

num_classes = config['dataset']['num_classes']
top_k = config['testing']['top_k']
in_channels = config['model']['in_channels']
nms_sigma = config['testing']['nms_sigma']
clip_length = config['dataset']['testing']['clip_length']
stride = config['dataset']['testing']['clip_stride']


def eval_one_id(net, data, sub, K):
    score_func = nn.Softmax(dim=-1)
    resize_crop = videotransforms.Resize(config['dataset']['testing']['crop_size'])

    sample_count = int(sub.split('_')[-1].split('.')[0])
    if sample_count < clip_length:
        offsetlist = [0]
    else:
        offsetlist = list(range(0, sample_count - clip_length + 1, stride))
        if (sample_count - clip_length) % stride:
            offsetlist += [sample_count - clip_length]

    data = np.transpose(data, [3, 0, 1, 2])
    data = resize_crop(data)
    data = torch.from_numpy(data)

    output = []
    for cl in range(num_classes):
        output.append([])
    res = torch.zeros(num_classes, top_k, 3)

    # print(video_name)
    for offset in offsetlist:
        clip = data[:, offset: offset + clip_length]
        clip = clip.float()
        clip = (clip / 255.0) * 2.0 - 1.0
        # clip = torch.from_numpy(clip).float()
        if clip.size(1) < clip_length:
            tmp = torch.zeros([clip.size(0), clip_length - clip.size(1),
                               96, 96]).float()
            clip = torch.cat([clip, tmp], dim=1)
        clip = clip.unsqueeze(0).cuda()
        with torch.no_grad():
            output_dict = net(clip)

        loc, conf, priors = output_dict['loc'], output_dict['conf'], output_dict['priors']
        prop_loc, prop_conf = output_dict['prop_loc'], output_dict['prop_conf']
        center = output_dict['center']
        loc = loc[0]
        conf = conf[0]
        prop_loc = prop_loc[0]
        prop_conf = prop_conf[0]
        center = center[0]

        pre_loc_w = loc[:, :1] + loc[:, 1:]
        loc = 0.5 * pre_loc_w * prop_loc + loc
        decoded_segments = torch.cat(
            [priors[:, :1] * clip_length - loc[:, :1],
             priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
        decoded_segments.clamp_(min=0, max=clip_length)

        conf = score_func(conf)
        prop_conf = score_func(prop_conf)
        center = center.sigmoid()

        conf = (conf + prop_conf) / 2.0
        conf = conf * center
        conf = conf.view(-1, num_classes).transpose(1, 0)
        conf_scores = conf.clone()

        for cl in range(1, num_classes):
            c_mask = conf_scores[cl] > 0.01
            scores = conf_scores[cl][c_mask]
            if scores.size(0) == 0:
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
            segments = decoded_segments[l_mask].view(-1, 2)
            # decode to original time
            # segments = (segments * clip_length + offset) / sample_fps
            segments = (segments + offset)  # / sample_fps
            segments = torch.cat([segments, scores.unsqueeze(1)], -1)

            output[cl].append(segments)
            # np.set_printoptions(precision=3, suppress=True)
            # print(idx_to_class[cl], tmp.detach().cpu().numpy())

    # print(output[1][0].size(), output[2][0].size())
    sum_count = 0
    for cl in range(1, num_classes):
        if len(output[cl]) == 0:
            continue
        tmp = torch.cat(output[cl], 0)
        tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k)
        res[cl, :count] = tmp
        sum_count += count

    flt = res.contiguous().view(-1, 3)
    flt = flt.view(num_classes, -1, 3)
    proposal_list = []
    for cl in range(1, num_classes):
        tmp = flt[cl].contiguous()
        tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
        if tmp.size(0) == 0:
            continue
        tmp = tmp.detach().cpu().numpy()
        for i in range(tmp.shape[0]):
            start_time = max(0, float(tmp[i, 0]))
            end_time = min(sample_count, float(tmp[i, 1]))
            if end_time <= start_time or (end_time - start_time) < 0.5 * K or 6 * K < (end_time - start_time):
                continue
            proposal_list.append([int(tmp[i, 0]),
                                  int(tmp[i, 1]),
                                  float(tmp[i, 2])])
    while (True):
        result_merge = recursive_merge(proposal_list)
        if len(result_merge) != len(proposal_list):
            proposal_list = result_merge
        else:
            break
    return proposal_list

def refine_result(pre_list, best_threshold, top_k=20):
    nms_pre_list, count = softnms_v2(torch.tensor(pre_list), top_k=top_k, score_threshold=best_threshold)
    if len(nms_pre_list)>0:
        nms_pre_list = nms_pre_list.detach().cpu().numpy().tolist()
    count_list = []
    for ii in nms_pre_list:
        count = 0
        for jj in pre_list:
            iou = cal_IOU(ii, jj)
            if iou > 0.3:
                count += 1
        count_list.append(count)
    sorted_nums = sorted(enumerate(count_list), key=lambda x: x[1], reverse=True)
    idx = [i[0] for i in sorted_nums]
    counts = [i[1] for i in sorted_nums]
    meam_count = []
    for i in sorted_nums:
        if i[1] > 1:
            meam_count.append(i[1])
    meam_count = np.mean(counts)
    result = []
    # 1.threshold
    for i in range(len(nms_pre_list)):
        if counts[i] >= meam_count and nms_pre_list[idx[i]][2]>best_threshold:
            result.append(nms_pre_list[idx[i]])
    # 2.top
    if len(result) < 1:
        for i in range(len(nms_pre_list)):
            if counts[i] >= meam_count:
                result.append(nms_pre_list[idx[i]])
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--dataset', default='CAS', type=str, help="SAMM/CAS/SMIC")
    parser.add_argument('--root_path',
                        default='/media/sai/data1/datasets/face/emotion/MEGC2022_testSet',
                        type=str)
    parser.add_argument('--model_path', default='../weight', type=str)
    parser.add_argument('--img_size', default=96, type=int, help='gap frames')
    parser.add_argument('--mode', default='Micro', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    info_path = '%s/%s_%s/model_parameter.json' % (args.model_path, args.dataset, args.mode)
    with open(info_path) as json_file:
        parameter = json.load(json_file)
    parameter = eval(parameter)
    json_file.close()
    if args.dataset=='CAS' and args.mode=='Micro':K = 6
    elif args.dataset=='CAS' and args.mode=='Macro':K = 18
    elif args.dataset=='SAMM' and args.mode=='Micro':K = 37
    else:K = 174
    net = BDNet(in_channels=in_channels, training=False)
    uid_path = '%s/%s_%s/' % (args.model_path, args.dataset, args.mode)
    uidList = os.listdir(uid_path)
    results = []
    sub_path = '%s/%s/%s/npy' % (args.root_path, args.dataset, args.mode)
    subList = os.listdir(sub_path)
    for sub in subList:
        print('----------------------%s-----------------------' % (sub))
        pre_list = []
        data = np.load(os.path.join(sub_path, sub))
        for uid in uidList:
            if 'json' in uid: continue
            checkpoint_path = '%s/%s/best.ckpt' % (uid_path, uid)
            net.load_state_dict(torch.load(checkpoint_path))
            net.eval().cuda()
            proposal_list = eval_one_id(net, data, sub, K)
            if len(proposal_list) > 0:
                pre_list.extend(proposal_list)
        result = refine_result(pre_list, parameter['best_conf'])
        if args.dataset == 'CAS':
            id = sub.split('_')[0]
        else:
            id = '%s_%s' % (sub.split('_')[0], sub.split('_')[1])
        results.append({'id':id, 'result':result})
        print({'vid':id, 'result':result})
    pred_results = []
    for i in results:
        vid, preds = i['id'], i['result']
        for pred in preds:
            if args.mode == 'Micro':
                type='me'
            else:type='mae'
            if args.dataset == 'CAS':
                dataset='cas'
            else:dataset='samm'
            pred_results.append({'vid':vid, 'pred_onset':int(pred[0]), 'pred_offset':int(pred[1]), 'type':type})
    with open('../submission/%s_pred_%s.csv' % (dataset, type), 'w', encoding='utf-8-sig') as ff:
        headers = ['vid', 'pred_onset', 'pred_offset', 'type']
        f_scv = csv.DictWriter(ff, headers)
        f_scv.writeheader()
        f_scv.writerows(np.array(pred_results))
    ff.close()