import os
from params import *
from server_setting import *
import numpy as np


def human_state():
    d = np.load('result/human.npz')
    result = d['result']
    label = d['label']
    diff = np.abs(result - label).reshape(-1, JOINT_LEN, 3)
    sqr_sum = np.sum(np.power(diff, 2), 2)
    joints = np.sqrt(sqr_sum)
    body = np.mean(joints, axis=1)
    print body
    r = []
    from dataset.human36 import Human36V
    hm = Human36V(HM_PATH)
    print body.mean()
    l = hm.test_lengths
    start = 0
    for i in l:
        r.append(body[start:start+i])
        start = start+i
    fr = []
    num = []
    for i in range(0,30,2):
        num.append(r[i].size+r[i+1].size+r[i+30].size+r[i+31].size)
        fr.append((r[i].sum()+r[i+1].sum()+r[i+30].sum()+r[i+31].sum())/num[-1])
    s = 0
    sn = 0
    for i in range(15):
        s+=fr[i]*num[i]
        sn+=num[i]
        print '{:15} {}'.format(hm.test_subjects[0].actionName[i*2],fr[i])
    print '{:15} {}'.format('<Mean Error>', s/sn)


if __name__ == '__main__':
    human_state()

