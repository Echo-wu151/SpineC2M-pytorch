import numpy as np
import torch


def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)v


def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result


def hard_dice(output, target):
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    target = target[:, 0]
    # target is not one hot encoded, output is
    # target must be the CPU segemtnation, not tensor. output is pytorch tensor
    num_classes = output.shape[1]
    output = output.argmax(1)
    foreground_classes = np.arange(1, num_classes)
    all_tp = []
    all_fp = []
    all_fn = []
    all_fg_dc = []
    for s in range(target.shape[0]):
        tp = []
        fp = []
        fn = []
        for c in foreground_classes:
            t_is_c = target[s] == c
            o_is_c = output[s] == c
            t_is_not_c = target[s] != c
            o_is_not_c = output[s] != c
            tp.append(np.sum(o_is_c & t_is_c))
            fp.append(np.sum(o_is_c & t_is_not_c))
            fn.append(np.sum(o_is_not_c & t_is_c))
        foreground_dice = [2 * i / (2 * i + j + k + 1e-8)
                           for i, j, k in zip(tp, fp, fn)]
        all_tp.append(tp)
        all_fp.append(fp)
        all_fn.append(fn)
        all_fg_dc.append(foreground_dice)
    return all_fg_dc, all_tp, all_fp, all_fn
