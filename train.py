import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
import json
import sys
import time
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from AFSD.common.dataset import THUMOS_Dataset, get_video_info, \
    load_video_data, detection_collate, get_video_anno
from torch.utils.data import DataLoader
from AFSD.BDNet import BDNet
from AFSD.multisegment_loss import MultiSegmentLoss
from AFSD.common.config import config
from AFSD.data.utils import load_label
from AFSD.common import videotransforms
from AFSD.common.segment_utils import softnms_v2
from AFSD.common.find_result import ParameterOptimizationAll, recursive_merge

mode = config['dataset']['mode']
datatype = config['dataset']['datatype']

batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
max_epoch = config['training']['max_epoch']
num_classes = config['dataset']['num_classes']
focal_loss = config['training']['focal_loss']
random_seed = config['training']['random_seed']

conf_thresh = config['testing']['conf_thresh']
top_k = config['testing']['top_k']
nms_sigma = config['testing']['nms_sigma']
clip_length = config['dataset']['testing']['clip_length']
stride = config['dataset']['testing']['clip_stride']

resume = config['training']['resume']
config['training']['ssl'] = 0.1


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    set_seed(1 + worker_id)


def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states


def set_rng_state(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(states[3])


def save_model(epoch, model, optimizer):
    torch.save(model.module.state_dict(),
               os.path.join(checkpoint_path, 'best.ckpt'))
    # torch.save({'optimizer': optimizer.state_dict(),
    #             'state': get_rng_states()},
    #            os.path.join(train_state_path, 'best.ckpt'))


def resume_training(resume, model, optimizer):
    if resume > 0:
        model_path = os.path.join(checkpoint_path, 'best.ckpt')
        if os.path.exists(model_path):
            model.module.load_state_dict(torch.load(model_path))
        train_path = os.path.join(train_state_path, 'best.ckpt')
        if os.path.exists(train_path):
            state_dict = torch.load(train_path)
            optimizer.load_state_dict(state_dict['optimizer'])
            set_rng_state(state_dict['state'])


def calc_bce_loss(start, end, scores):
    start = torch.tanh(start).mean(-1)
    end = torch.tanh(end).mean(-1)
    loss_start = F.binary_cross_entropy(start.view(-1),
                                        scores[:, 0].contiguous().view(-1).cuda(),
                                        reduction='mean')
    loss_end = F.binary_cross_entropy(end.view(-1),
                                      scores[:, 1].contiguous().view(-1).cuda(),
                                      reduction='mean')
    return loss_start, loss_end


def forward_one_epoch(net, clips, targets, scores=None, training=True, ssl=True):
    clips = clips.cuda()
    targets = [t.cuda() for t in targets]

    if training:
        if ssl:
            output_dict = net(clips, proposals=targets, ssl=ssl)
        else:
            output_dict = net(clips, ssl=False)
    else:
        with torch.no_grad():
            output_dict = net(clips)

    if ssl:
        anchor, positive, negative = output_dict
        loss_ = []
        weights = [1, 0.1, 0.1]
        for i in range(3):
            loss_.append(nn.TripletMarginLoss()(anchor[i], positive[i], negative[i]) * weights[i])
        trip_loss = torch.stack(loss_).sum(0)
        return trip_loss
    else:
        loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct = CPD_Loss(
            [output_dict['loc'], output_dict['conf'],
             output_dict['prop_loc'], output_dict['prop_conf'],
             output_dict['center'], output_dict['priors']],
            targets)
        loss_start, loss_end = calc_bce_loss(output_dict['start'], output_dict['end'], scores)
        scores_ = F.interpolate(scores, scale_factor=1.0 / 4)
        loss_start_loc_prop, loss_end_loc_prop = calc_bce_loss(output_dict['start_loc_prop'],
                                                               output_dict['end_loc_prop'],
                                                               scores_)
        loss_start_conf_prop, loss_end_conf_prop = calc_bce_loss(output_dict['start_conf_prop'],
                                                                 output_dict['end_conf_prop'],
                                                                 scores_)
        loss_start = loss_start + 0.1 * (loss_start_loc_prop + loss_start_conf_prop)
        loss_end = loss_end + 0.1 * (loss_end_loc_prop + loss_end_conf_prop)
        return loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct, loss_start, loss_end


def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num):
    net.train()  # Set model to training mode
    training = True
    loss_loc_val = 0
    loss_conf_val = 0
    loss_prop_l_val = 0
    loss_prop_c_val = 0
    loss_ct_val = 0
    loss_start_val = 0
    loss_end_val = 0
    loss_trip_val = 0
    loss_contras_val = 0
    cost_val = 0
    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):
            time.sleep(0.15)
            loss_l, loss_c, loss_prop_l, loss_prop_c, \
            loss_ct, loss_start, loss_end = forward_one_epoch(
                net, clips, targets, scores, training=training, ssl=False)

            loss_l = loss_l * config['training']['lw']
            loss_c = loss_c * config['training']['cw']
            loss_prop_l = loss_prop_l * config['training']['lw']
            loss_prop_c = loss_prop_c * config['training']['cw']
            loss_ct = loss_ct * config['training']['cw']
            cost = loss_l + loss_c + loss_prop_l + loss_prop_c + loss_ct + loss_start + loss_end

            if flags[0]:
                loss_trip = forward_one_epoch(net, ssl_clips, ssl_targets, training=training,
                                              ssl=True)
                loss_trip *= config['training']['ssl']
                cost = cost + loss_trip
                loss_trip_val += loss_trip.cpu().detach().numpy()

            if training:
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

            loss_loc_val += loss_l.cpu().detach().numpy()
            loss_conf_val += loss_c.cpu().detach().numpy()
            loss_prop_l_val += loss_prop_l.cpu().detach().numpy()
            loss_prop_c_val += loss_prop_c.cpu().detach().numpy()
            loss_ct_val += loss_ct.cpu().detach().numpy()
            loss_start_val += loss_start.cpu().detach().numpy()
            loss_end_val += loss_end.cpu().detach().numpy()
            cost_val += cost.cpu().detach().numpy()
            pbar.set_postfix(loss='{:.5f}'.format(float(cost.cpu().detach().numpy())))

    loss_loc_val /= (n_iter + 1)
    loss_conf_val /= (n_iter + 1)
    loss_prop_l_val /= (n_iter + 1)
    loss_prop_c_val /= (n_iter + 1)
    loss_ct_val /= (n_iter + 1)
    loss_start_val /= (n_iter + 1)
    loss_end_val /= (n_iter + 1)
    loss_trip_val /= (n_iter + 1)
    cost_val /= (n_iter + 1)

    if training:
        prefix = 'Train'
    else:
        prefix = 'Val'
    # save_model(epoch, net, optimizer)

    plog = 'Epoch-{} {} Loss: Total - {:.5f}, loc - {:.5f}, conf - {:.5f}, ' \
           'prop_loc - {:.5f}, prop_conf - {:.5f}, ' \
           'IoU - {:.5f}, start - {:.5f}, end - {:.5f}'.format(
        i, prefix, cost_val, loss_loc_val, loss_conf_val, loss_prop_l_val, loss_prop_c_val,
        loss_ct_val, loss_start_val, loss_end_val
    )
    plog = plog + ', Triplet - {:.5f}'.format(loss_trip_val)
    print(plog)
    return net, optimizer


def eval_one_id(net, uid):
    net.eval().cuda()
    score_func = nn.Softmax(dim=-1)
    resize_crop = videotransforms.Resize(config['dataset']['testing']['crop_size'])

    test_info_path = '%s/%s_val.json' % (config['dataset']['testing']['video_info_path'], uid)
    with open(test_info_path) as json_file:
        test_info = json.load(json_file)
    json_file.close()

    one_pre = []
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
        data = np.load(data_path)
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
                c_mask = conf_scores[cl] > conf_thresh
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

        sum_count = min(sum_count, top_k)
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
                if end_time <= start_time:
                    continue
                proposal_list.append([float(tmp[i, 0]),
                                      float(tmp[i, 1]),
                                      float(tmp[i, 2])])
        # while (True):
        #     result_merge = recursive_merge(proposal_list)
        #     if len(result_merge) != len(proposal_list):
        #         proposal_list = result_merge
        #     else:
        #         break
        one_pre.append({'label': annotations, 'result': proposal_list})
    return one_pre

def init_model():
    """
                Setup model
                """
    net = BDNet(in_channels=config['model']['in_channels'],
                backbone_model=config['model']['backbone_model'])
    # net = nn.DataParallel(net, device_ids=[0]).cuda()
    net = nn.DataParallel(net, device_ids=list(range(config['dataset']['training']['ngpu']))).cuda()
    """
    Setup optimizer
    """
    # optimizer = torch.optim.SGD(net.parameters(),
    #                          lr=learning_rate,
    #                          momentum=0.9,
    #                          weight_decay=weight_decay)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate, weight_decay=weight_decay)  #
    """
    Setup loss
    """
    piou = config['training']['piou']
    CPD_Loss = MultiSegmentLoss(num_classes, piou, 1.0, use_focal_loss=focal_loss)
    return net.cuda(), optimizer, CPD_Loss

if __name__ == '__main__':
    set_seed(random_seed)
    logger.add('log/%s_%s.log' % (datatype, mode), encoding='utf-8')

    logger.info(config)


    """
    Setup dataloader
    """

    path_xlsx = '../datasets/%s.xlsx' % (datatype)

    _, _, labels, paths, all_subjects = load_label(path_xlsx, datatype, mode)

    all_pres = []
    # train
    count = 0
    for uid in all_subjects:
        print(uid)
        net, optimizer, CPD_Loss = init_model()
        checkpoint_path = '%s/%s_%s/%s' % (config['training']['checkpoint_path'], datatype, mode, uid)
        # train_state_path = '%s/%s_%s/training/%s' % (config['training']['checkpoint_path'], datatype, mode, uid)
        # train_state_path = os.path.join(checkpoint_path, 'training')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        # if not os.path.exists(train_state_path):
        #     os.makedirs(train_state_path)

        video_info_path = '%s/%s_info.csv' % (config['dataset']['training']['video_info_path'], uid)
        video_anno_path = '%s/%s_annotations.csv' % (config['dataset']['training']['video_anno_path'], uid)
        train_video_infos = get_video_info(video_info_path)
        train_video_annos = get_video_anno(train_video_infos, video_anno_path)
        train_data_dict = load_video_data(train_video_infos,
                                          config['dataset']['training']['video_data_path'])
        train_dataset = THUMOS_Dataset(train_data_dict,
                                       train_video_infos,
                                       train_video_annos)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=0, worker_init_fn=worker_init_fn,
                                       collate_fn=detection_collate, pin_memory=True, drop_last=True)
        epoch_step_num = len(train_dataset) // batch_size

        """
        Start training
        """

        import copy
        best_model_wts = copy.deepcopy(net.state_dict())
        resume_training(resume, net, optimizer)
        iterNum, best_fscore, best_one_pre = 0, 0, []
        for i in range(max_epoch + 1):
            net, optimizer = run_one_epoch(i, net, optimizer, train_data_loader, epoch_step_num)
            one_pre = eval_one_id(net, uid)
            all_pre = all_pres.copy()
            all_pre.extend(one_pre)
            result = ParameterOptimizationAll(all_pre)
            save_model(i, net, optimizer)
            fscore = result['f1_score']
            if best_fscore < fscore:
                iterNum = 0
                best_fscore = fscore
                best_one_pre = one_pre
                save_model(i, net, optimizer)
                best_model_wts = copy.deepcopy(net.state_dict())
                logger.info('----------------%s: Epoch: %d Parameter: %s ----------------' % (uid, i, result))
            else:
                iterNum += 1
                if iterNum > 4: break
                net.load_state_dict(best_model_wts)
                print('******uid:{} epoch:{} fscore:{}******'.format(uid, i, result))
        all_pres.extend(best_one_pre)
