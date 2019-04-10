import os
from params import *
from server_setting import *
import numpy as np
import pa

use_pa = False

def human_state():
    # You need to change the path, Remember to change testset for S9 in protocal 1 !!!
    d = np.load('result/human_p2_c12.npz')
    result = d['result']
    label = d['label']
    # if you need to test the 17 keypoints, use the first two row below.
    #result = result[:,[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27],:]
    #label = label[:,[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27],:]

    # if you need to test the 14 keypoints(without 0,12,13)th keypoint
    #result = result[:,[1,2,3,6,7,8,14,15,17,18,19,25,26,27],:]
    #label = label[:,[1,2,3,6,7,8,14,15,17,18,19,25,26,27],:]

    if use_pa:
        for i in range(result.shape[0]):
             gt = np.reshape(label[i], [-1, 3])
             out = np.reshape(result[i], [-1, 3])
             _, Z, T, b, c = pa.compute_similarity_transform(gt, out, compute_optimal_scale=True)
             out = (b * out.dot(T)) + c
             label[i] = out


    diff = np.abs(result - label).reshape(-1, JOINT_LEN, 3)
    # if you need to test the 14 keypoints(without 0,12,13)th keypoint
    diff = diff[:,[1,2,3,6,7,8,14,15,17,18,19,25,26,27],:]

    sqr_sum = np.sum(np.power(diff, 2), 2)
    joints = np.sqrt(sqr_sum)
    body = np.mean(joints, axis=1)
    print body
    r = []
    from dataset.human36rgb import Human36RGBV
    hm = Human36RGBV(HM_RGB_PATH)
    print body.mean()
    l = hm.test_lengths
    start = 0
    print l
    for i in l:
        r.append(body[start:start+i])
        start = start+i
    fr = []
    num = []

    # This forloop is used for protocal 2
    for i in range(0,30,2):
        num.append(r[i].size + r[i + 1].size + r[i + 30].size + r[i + 31].size)
        fr.append((r[i].sum() + r[i + 1].sum() + r[i + 30].sum() + r[i + 31].sum()) / num[-1])
    s = 0
    sn = 0
    print num
    print fr
    for i in range(15):
        s+=fr[i]*num[i]
        sn+=num[i]
        print '{:15} {}'.format(hm.test_subjects[0].actionName[i*2],fr[i])
    print '{:15} {}'.format('<Mean Error>', s/sn)


if __name__ == '__main__':
    human_state()
'''     
    # delete these three bad sequence:Greeting,SittingDown1,Waiting1 ; If you do not delete them,just use the else: part.
    for i in range(0,30,2):
        if i == 6:
            num.append(r[i + 1].size + r[i + 30].size + r[i + 31].size)
            fr.append((r[i+1].sum()+r[i+30].sum()+r[i+31].sum())/num[-1])
        elif i == 18:
            num.append(r[i].size + r[i + 30].size + r[i + 31].size)
            fr.append((r[i].sum() + r[i + 30].sum() + r[i + 31].sum()) / num[-1])
        elif i == 22:
            num.append(r[i].size + r[i + 30].size + r[i + 31].size)
            fr.append((r[i].sum() + r[i + 30].sum() + r[i + 31].sum()) / num[-1])
        else:
            num.append(r[i].size+r[i+1].size+r[i+30].size+r[i+31].size)
            fr.append((r[i].sum()+r[i+1].sum()+r[i+30].sum()+r[i+31].sum())/num[-1])

    # This forloop is used for protocal 1
    for i in range(0,30,2):
        num.append(r[i].size+r[i+1].size)
        fr.append((r[i].sum()+r[i+1].sum())/num[-1])'''


