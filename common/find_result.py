# -*- coding: utf-8 -*-
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(__file__))
from calculate import cal_f1_score, spotting_evaluation_V2


def softnms_v2(segments, sigma=0.5, top_k=50, score_threshold=0.1):
    import torch
    segments = torch.tensor(segments)
    segments = segments.cpu()
    new_segments = []
    if segments.size()[0] < 1:
        return new_segments, 0
    tstart = segments[:, 0]
    tend = segments[:, 1]
    tscore = segments[:, 2]
    done_mask = tscore < -1  # set all to False
    undone_mask = tscore >= score_threshold
    while undone_mask.sum() > 1 and done_mask.sum() < top_k:
        idx = tscore[undone_mask].argmax()
        idx = undone_mask.nonzero()[idx].item()

        undone_mask[idx] = False
        done_mask[idx] = True

        top_start = tstart[idx]
        top_end = tend[idx]
        _tstart = tstart[undone_mask]
        _tend = tend[undone_mask]
        tt1 = _tstart.clamp(min=top_start)
        tt2 = _tend.clamp(max=top_end)
        intersection = torch.clamp(tt2 - tt1, min=0)
        duration = _tend - _tstart
        tmp_width = torch.clamp(top_end - top_start, min=1e-5)
        iou = intersection / (tmp_width + duration - intersection)
        scales = torch.exp(-iou ** 2 / sigma)
        tscore[undone_mask] *= scales
        undone_mask[tscore < score_threshold] = False
    count = done_mask.sum()
    new_segments = torch.stack([tstart[done_mask], tend[done_mask], tscore[done_mask]], -1)
    # segments = segments.detach().cpu().numpy().tolist()
    # for segment in segments:
    #     new_segments.append([int(segment[0]), int(segment[1]), segment[2]])
    return new_segments, count


def recursive_merge(inter, start_index=0):
    for i in range(start_index, len(inter) - 1):
        if inter[i][1] >= inter[i + 1][0]:
            # new_start = int(0.5 * (inter[i][0] + inter[i + 1][0]))
            # new_end = int(0.5 * (inter[i][1] + inter[i + 1][1]))
            # new_score = max(inter[i][2], inter[i + 1][2])
            new_start = int(
                inter[i][0] * inter[i][2] / (inter[i][2] + inter[i + 1][2]) + inter[i + 1][0] * inter[i + 1][2] / (
                            inter[i][2] + inter[i + 1][2]))
            new_end = int(
                inter[i][1] * inter[i][2] / (inter[i][2] + inter[i + 1][2]) + inter[i + 1][1] * inter[i + 1][2] / (
                            inter[i][2] + inter[i + 1][2]))
            new_score = max(inter[i][2], inter[i + 1][2])
            inter[i] = [new_start, new_end, new_score]
            del inter[i + 1]
            return recursive_merge(inter.copy(), start_index=i)
    return inter


def ParameterOptimization(all_pre, sigma=0.5, top_k=20, score_threshold=0.01):
    one_subject_TP, one_subject_FP, one_subject_FN = 0, 0, 0
    for i in range(len(all_pre)):
        result, label = all_pre[i]['result'], all_pre[i]['label']
        result_, count_ = softnms_v2(result, sigma=sigma, top_k=top_k, score_threshold=score_threshold)
        score_threshold = 0
        if len(result_) > 0:
            result_ = result_.detach().cpu().numpy().tolist()
            for ii in result_:
                score_threshold += ii[2]
            score_threshold /= len(result_)
        result, count = softnms_v2(result_, sigma=sigma, top_k=len(result_), score_threshold=score_threshold)
        if len(result) > 0:
            result = result.detach().cpu().numpy().tolist()
        while (True):
            result_merge = recursive_merge(result)
            if len(result_merge) != len(result):
                result = result_merge
            else:
                break
        segments = []
        for j in label:
            segments.append(j['segment'])
        if len(result) < 1:
            TP, FP, FN = 0, 0, len(label)
        else:
            TP, FP, FN = spotting_evaluation_V2(result, segments)
        one_subject_TP += TP
        one_subject_FP += FP
        one_subject_FN += FN
    if one_subject_TP + one_subject_FP == 0:
        f1_score = 0
    else:
        recall, precision, f1_score = cal_f1_score(one_subject_TP, one_subject_FP,
                                                   one_subject_FN)

    return {'f1_score': f1_score, 'one_subject_TP': one_subject_TP, 'one_subject_FP': one_subject_FP,
            'one_subject_FN': one_subject_FN}


def ParameterOptimizationAll(all_pre):
    best_f1score, best_conf, best_TP, best_FP, best_FN = 0, 0, 0, 0, 0
    for conf in np.linspace(0.1, 1, num=19):
        one_subject_TP, one_subject_FP, one_subject_FN = 0, 0, 0
        for i in range(len(all_pre)):
            result, label = all_pre[i]['result'], all_pre[i]['label']
            result, count = softnms_v2(result, top_k=len(result), score_threshold=conf)
            if len(result) > 0:
                result = result.detach().cpu().numpy().tolist()
            while (True):
                result_merge = recursive_merge(result)
                if len(result_merge) != len(result):
                    result = result_merge
                else:
                    break
            segments = []
            for j in label:
                segments.append(j['segment'])
            if len(result) < 1:
                TP, FP, FN = 0, 0, len(label)
            else:
                TP, FP, FN = spotting_evaluation_V2(result, segments)
            one_subject_TP += TP
            one_subject_FP += FP
            one_subject_FN += FN
        if one_subject_TP + one_subject_FP == 0:
            f1_score = 0
        else:
            recall, precision, f1_score = cal_f1_score(one_subject_TP, one_subject_FP,
                                                       one_subject_FN)
        # print('f1_score', f1_score, 'conf', conf, one_subject_TP, one_subject_FP, one_subject_FN)
        if best_f1score < f1_score:
            best_f1score, best_conf, best_TP, best_FP, best_FN = f1_score, conf, one_subject_TP, one_subject_FP, one_subject_FN

    return {'f1_score': best_f1score, 'best_conf': best_conf, 'best_TP': best_TP,
            'best_FP': best_FP,
            'best_FN': best_FN}

