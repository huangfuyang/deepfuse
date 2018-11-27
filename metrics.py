import numpy as np
from params import *
import torch


def accuracy_portion(output, target, t=ERROR_THRESH):
    diff = np.abs(output - target).reshape(-1,3)
    sqr_sum = np.sum(np.square(diff),axis=1)
    out = np.zeros(sqr_sum.size)
    t = t**2
    out[sqr_sum < t] = 1
    good = np.sum(out) / out.size
    return good * 100


def accuracy_error_thresh_portion_batch(output, target, t=ERROR_THRESH):
    batch_size = target.size(0)
    sample_size = target.size(1)
    diff = torch.abs(output-target).view(batch_size,-1,3)
    sqr_sum = torch.sum(torch.pow(diff,2),2)
    out = torch.zeros(sqr_sum.size())
    t = t**2
    out[sqr_sum<t] = 1
    good = torch.sum(out)/(out.size(1)*batch_size)
    return good*100


def good_frame(output, target, t = ERROR_THRESH):
    batch_size = target.size(0)
    sample_size = target.size(1)
    diff = torch.abs(output - target).view(batch_size, -1, 3)
    sqr_sum = torch.sum(torch.pow(diff, 2), 2)
    out = torch.zeros(sqr_sum.size())
    t = t ** 2
    out[sqr_sum > t] = 1
    out = torch.sum(out, 1)
    out[out>0] = 1
    good = 1-torch.sum(out) / batch_size
    return good * 100


def mean_error(output,target):
    batch_size = target.size(0)
    diff = torch.abs(output - target).view(batch_size, JOINT_LEN, 3)
    sqr_sum = torch.sum(torch.pow(diff, 2), 2)
    sqrt_row = torch.sqrt(sqr_sum)
    sqrt_row = sqrt_row.mean(dim=0)
    # print sqrt_row
    # print sqrt_row[10],sqrt_row[14],sqrt_row[17],sqrt_row[20],sqrt_row[6]

    # print (sqrt_row)
    if batch_size != 0:
        return torch.mean(sqrt_row),sqrt_row
    else:
        return 0


def mean_error_heatmap(output,target):
    batch_size = target.size(0)
    diff = torch.abs(output - target).view(batch_size, -1, 2)
    sqr_sum = torch.sum(torch.pow(diff, 2), 2)
    sqrt_row = torch.sqrt(sqr_sum)
    if batch_size == 1:
        print (sqrt_row)
    if batch_size != 0:
        return torch.mean(sqrt_row)
        # return torch.mean(sqrt_row[:,4])
    else:
        return 0



def mean_error_heatmap_topk(output,target):
    batch_size = target.size(0)
    joint_len = JOINT_LEN
    err = 99999
    k = output.size()[2]
    target = target.view(batch_size,-1,2).unsqueeze(2).repeat(1,1,k,1)
    r, indice = (output-target).pow(2).sum(-1).min(-1)
    r = r.sqrt()
    ind = torch.FloatTensor(batch_size,joint_len,2)
    for i in range(batch_size):
        for j in range(joint_len):
            ind[i][j] = output[i][j][indice[i][j]].view(-1)
    # if batch_size == 1:
    #     print r
    if batch_size != 0:
        return torch.mean(r,0), ind
        # return torch.mean(sqrt_row[:,4])
    else:
        return 0


def mean_error_heatmap3d(output,target):
    batch_size = target.size(0)
    diff = (output - target).abs().view(batch_size, -1, 3)
    # print diff
    sqrt_row = diff.pow(2).sum(2).sqrt()
    if batch_size == 1:
        print (sqrt_row)
    if batch_size != 0:
        return torch.mean(sqrt_row)
        # return torch.mean(sqrt_row[:,4])
    else:
        return 0


def mean_error_heatmap3d_topk(output,target):
    batch_size = target.size(0)
    joint_len = JOINT_LEN
    err = 99999
    k = output.size()[2]
    target = target.view(batch_size,-1,3).unsqueeze(2).repeat(1,1,k,1)
    r, indice = (output-target).pow(2).sum(-1).min(-1)
    r = r.sqrt()
    ind = torch.FloatTensor(batch_size,joint_len,3)
    for i in range(batch_size):
        for j in range(joint_len):
            ind[i][j] = output[i][j][indice[i][j]].view(-1)
    # if batch_size == 1:
    #     print r
    # print r
    if batch_size != 0:
        return torch.mean(r)
        # return torch.mean(sqrt_row[:,4])
    else:
        return 0
