import torch
import torch.nn as nn
import os
import numpy as np
import argparse
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from AFSD.common import videotransforms
from AFSD.BDNet import BDNet
from AFSD.common.segment_utils import softnms_v2
from AFSD.common.config import config
from AFSD.data.utils import *
from AFSD.common.find_result import ParameterOptimizationAll

num_classes = config['dataset']['num_classes']
# conf_thresh = config['testing']['conf_thresh']
top_k = config['testing']['top_k']
nms_sigma = config['testing']['nms_sigma']
clip_length = config['dataset']['testing']['clip_length']
stride = config['dataset']['testing']['clip_stride']
# checkpoint_path = config['testing']['checkpoint_path']



def eval_one_id(net, uid, K):
    test_info_path = '%s/%s_val.json' % (config['dataset']['testing']['video_info_path'], uid)
    with open(test_info_path) as json_file:
        test_info = json.load(json_file)
    json_file.close()

    one_pre = []
    checkpoint_path = '%s/%s_%s/%s/best.ckpt' % (
    config['training']['checkpoint_path'], config['dataset']['datatype'], config['dataset']['mode'], uid)
    if os.path.exists(checkpoint_path):
        net.load_state_dict(torch.load(checkpoint_path))
        net.eval().cuda()
        score_func = nn.Softmax(dim=-1)
        resize_crop = videotransforms.Resize(config['dataset']['testing']['crop_size'])

        for video_id in test_info:
            annotations = test_info[video_id]['annotations']
            sample_fps, sample_count = test_info[video_id]['sample_fps'], test_info[video_id]['sample_count']
            if sample_count < clip_length:
                offsetlist = [0]
            else:
                offsetlist = list(range(0, sample_count - clip_length + 1, stride))
                if (sample_count - clip_length) % stride:
                    offsetlist += [sample_count - clip_length]

            data_path = '%s/%s.npy' % (config['dataset']['testing']['video_data_path'], video_id)
            data = np.load(data_path, allow_pickle=True)
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
                    segments = (segments + offset)# / sample_fps
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

            # sum_count = min(sum_count, top_k)
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
                    if end_time <= start_time or (end_time-start_time)<0.5*K or 7*K<(end_time-start_time):
                        continue
                    proposal_list.append([float(tmp[i, 0]),
                                          float(tmp[i, 1]),
                                          float(tmp[i, 2])])
            # proposal_list = recursive_merge(proposal_list)
            one_pre.append({'label': annotations, 'result': proposal_list})
    else:
        for video_id in test_info:
            annotations = test_info[video_id]['annotations']
            one_pre.append({'label': annotations, 'result': []})
    return one_pre


if __name__ == '__main__':
    path_xlsx = '../datasets/%s.xlsx' % (config['dataset']['datatype'])
    if config['dataset']['datatype'] == 'CAS':
       _, _, labels, paths = read_xlsx_cas(path_xlsx, config['dataset']['mode'])
    elif config['dataset']['datatype'] == 'SAMM':
        _, _, labels, paths = read_xlsx_samm(path_xlsx, config['dataset']['mode'])
    else:
        _, _, labels, paths = read_xlsx_smic(path_xlsx, config['dataset']['mode'])
    K = cal_k(labels)

    _, _, labels, paths, all_subjects = load_label(path_xlsx, config['dataset']['datatype'], config['dataset']['mode'])

    net = BDNet(in_channels=config['model']['in_channels'],
                training=False)
    all_pres = []
    for uid in all_subjects[:2]:
        print(uid)
        one_pre = eval_one_id(net, uid, K)
        all_pres.extend(one_pre)
    result = ParameterOptimizationAll(all_pres)
    print(result)

