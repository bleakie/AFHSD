import numpy as np
from scipy import signal

def cal_IOU(interval_1, interval_2):
    intersection = [max(interval_1[0], interval_2[0]), min(interval_1[1], interval_2[1])]
    union_set    = [min(interval_1[0], interval_2[0]), max(interval_1[1], interval_2[1])]
    if intersection[0]<=intersection[1]:
        len_inter = intersection[1]-intersection[0]+1
        len_union = union_set[1]-union_set[0]+1
        return len_inter/len_union
    else:
        return 0


def cal_TP(left_count_1, label):
    result = []
    for inter_2 in label:
        temp = 0
        for inter_1 in left_count_1:
            if cal_IOU(inter_1, inter_2)>=0.5:
                temp += 1
        result.append(temp)
    return result

def spotting_evaluation(pred, express_inter, K, P):
    pred = np.array(pred)
    threshold = np.mean(pred)+ P*(np.max(pred)-np.mean(pred))
    num_peak = signal.find_peaks(pred, height=threshold, distance=K*2)
    pred_inter = []
    
    for peak in num_peak[0]:
        pred_inter.append([peak-K, peak+K])

    result = cal_TP(pred_inter, express_inter)
    result = np.array(result)
    TP = len(np.where(result!=0)[0])
    n = len(pred_inter)-(sum(result)-TP)
    m = len(express_inter)
    FP = n-TP
    FN = m-TP

    return TP, FP, FN, pred_inter

def spotting_evaluation_V2(pred_inter, express_inter):
    result = cal_TP(pred_inter, express_inter)
    result = np.array(result)
    TP = len(np.where(result!=0)[0])
    n = len(pred_inter)-(sum(result)-TP)
    m = len(express_inter)
    FP = n-TP
    FN = m-TP

    return TP, FP, FN

def cal_f1_score(TP, FP, FN):
    if TP==0:
        recall, precision, f1_score = 0, 0, 0
    else:
        recall = TP/(TP+FP)
        precision = TP/(TP+FN)
        f1_score = 2*recall*precision/(recall+precision)
    return recall, precision, f1_score