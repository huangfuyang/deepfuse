from torch.utils.data import Dataset,DataLoader
import numpy as np
import os,glob
import skvideo.io
from params import *
from server_setting import *
from mcv import make_mcv
from camera_hm import *
from time import time
from tempfile import TemporaryFile
import glob
import pandas as pd
import quaternion
from time import time
from augmentation import data_augmentation3D,random_shut
from helper import timeit
import h5py


class MyReader():
    """
    video reader struct
    """
    def __init__(self):
        self.current_video = ''
        self.readers = None
        self.current_frame = 0


class Subject(object):
    """
    Encapsulation class for subject
    """
    def __init__(self, name, root=HM_PATH):
        """
        :param name: name for subject
        :param root: dataset root path
        """
        self.name = name
        self.actionName = []
        self.videoName = []
        self.pvhName = []
        self.groundTruth = []
        self.gtName = []
        self.actionDict = {}
        self._bspath = os.path.join(root, name, 'MySegmentsMat', 'ground_truth_bs')
        # self._gtpath = os.path.join(root, 'h36m', name, 'MyPoses', '3D_positions')
        self._gtpath = os.path.join(root, name, 'MySegmentsMat', 'ground_truth_position')
        self.import_data()
        # self.ActionGroup()

    def import_data(self):
        """
        init subject and action
        :return:
        """
        p = os.path.join(HM_PATH,self.name, 'MySegmentsMat', 'ground_truth_bs','*.mp4')
        videos = glob.glob(p)
        for file in videos:
            self.videoName.append(file.split('/')[-1])

        # print Subject.ActionName
        p = os.path.join(HM_PATH, self.name, 'MySegmentsMat', 'ground_truth_position', '*.csv')
        csvs = glob.glob(p)
        for file in csvs:
            self.gtName.append(file.split('/')[-1])

        files = os.listdir(os.path.join(HM_PATH,self.name, 'MySegmentsMat', 'ground_truth_bs'))
        for name in files:
            if os.path.isdir(os.path.join(HM_PATH,self.name, 'MySegmentsMat', 'ground_truth_bs',name)):
                self.pvhName.append(name)

        self.videoName.sort()
        self.gtName.sort()
        self.pvhName.sort()
        # print self.videoName
        # print self.gtName
        for f in self.videoName:
            aname = f.split('.')[0]
            if aname not in self.actionDict:
                self.actionDict[aname] = [aname+'.csv']
            self.actionDict[aname].append(f)
        if len(self.videoName) == 0:
            self.actionName = self.pvhName
        else:
            self.actionName = self.actionDict.keys()
            self.actionName.sort()
        # print self.actionName
        # print self.actionDict[self.actionName[0]]

    def select_action(self, num):
        """
        select action including all the 4 video path and 1 gt
        :param num: action name or action index
        :return:
        """
        # print action
        if type(num) is int:
            actionName = self.actionName[num]
        else:
            actionName = num
        gtfile = self.actionDict[actionName][0]
        gtfile = os.path.join(self._gtpath,gtfile)
        gt = np.loadtxt(gtfile,dtype=np.str, delimiter=',')[:,:-1].astype(np.float32)
        # with h5py.File(gtfile, 'r') as hf:
        #     points = hf[hf.keys()[0]][()]
        # return [os.path.join(self._bspath,x) for x in self.actionDict[actionName][1:]], points.T
        return [os.path.join(self._bspath,x) for x in self.actionDict[actionName][1:]], gt

    def __str__(self):
        return self.name


class Human36(Dataset):
    """
    human3.6M dataset for raw matte data. preprocess raw data into volume
    """
    def __init__(self,root_path):
        """
        :param root_path:dataset root path
        """

        self.root_path = root_path
        self.current_video = ""
        self.subjects = filter(lambda x: os.path.isdir(os.path.join(root_path, x)), os.listdir(root_path))
        self.subjects.sort()
        # self.subjects = ['S1','S5','S6','S7','S8','S9','S11']
        self.subjects = ['S1']
        self.subjects = [Subject(i) for i in self.subjects]

        self.data_dict = {}
        self.data = []
        self.length = 0
        self.video_readers = [MyReader() for i in range(NUM_CAM_HM)]
        self.frame_data = [None]*NUM_CAM_HM
        self.save = False
        self.raw = False
        # self.subjects[0].actionName = ['Directions']
        self.cams = load_cameras(os.path.join(HM_PATH,'h36m','cameras.h5'))
        for sub in self.subjects:
            self.data_dict[sub.name] = {}
            # self.data_dict[sub]['camera'] = self.cams[sub]
            for act in sub.actionName:
                self.data_dict[sub.name][act] = {}
                print 'load',sub,act
                videos, label = sub.select_action(act)
                lines = label.shape[0]
                self.data_dict[sub.name][act]['label'] = label
                self.data_dict[sub.name][act]['video'] = videos
                for i in range(lines):
                    self.data.append([sub.name, act, i])
                self.length += lines
        print self.length,'data loaded'
        # load camera params
        # self.subjects_length.append(sub_len)

    def __getitem__(self, item):
        """
        get specific frame
        :param item: index of frame
        :return:
        """
        info = self.data[item]
        # info = [sub.name, act, i]
        data = self.data_dict[info[0]][info[1]]
        index = info[2]
        videos = data['video']
        label = data['label'][index]
        d_path = os.path.join(os.sep, os.path.join(*videos[0].split(os.sep)[1:-1]),info[1])
        np_path = os.path.join(d_path, str(index).zfill(5)+'.npz')
        if not self.raw and os.path.isfile(np_path):
            npz = np.load(np_path)
            mcv, label, mid, leng = npz['mcv'], npz['label'], npz['mid'], npz['len']
            return mcv, label, mid, leng.reshape((1))
        for i in range(NUM_CAM_HM):
            # current video reader not matched
            if self.video_readers[i].current_video != videos[i]:
                self.video_readers[i].current_video = videos[i]
                inputparameters = {}
                outputparameters = {}
                outputparameters['-pix_fmt'] = 'gray'
                if self.video_readers[i].readers is skvideo.io.FFmpegReader:
                    self.video_readers[i].readers.close()
                self.video_readers[i].readers = skvideo.io.FFmpegReader(self.video_readers[i].current_video,
                                                 inputdict=inputparameters,
                                                 outputdict=outputparameters)
                print 'load',self.video_readers[i].current_video,self.video_readers[i].readers.inputframenum,'frames'
                if index<0 or index >= self.video_readers[i].readers.inputframenum:
                    print 'frame overloaded ',videos[0], index, self.video_readers[i].readers.inputframenum
                    return None
                    # raise ValueError("index not valid")
                self.video_readers[i].current_frame = 0
                for frame in self.video_readers[i].readers.nextFrame():
                    # determine frame number
                    if self.video_readers[i].current_frame == index:
                        self.video_readers[i].current_frame+=1
                        self.frame_data[i] = frame
                        break
                    else:
                        self.video_readers[i].current_frame+=1
            # video matched
            else:
                if index < self.video_readers[i].current_frame or index >= self.video_readers[i].readers.inputframenum:
                    print videos[0],index,self.video_readers[i].current_frame,self.video_readers[i].readers.inputframenum
                    return None
                    # raise ValueError("index not valid")
                for frame in self.video_readers[i].readers.nextFrame():
                    # determine frame number
                    if self.video_readers[i].current_frame == index:
                        self.video_readers[i].current_frame += 1
                        self.frame_data[i] = frame
                        break
                    else:
                        self.video_readers[i].current_frame += 1
        # make mcv
        # t = time
        if self.raw:
            return self.frame_data,label
        mcv,mid,leng = make_mcv(self.frame_data, label, self.cams[info[0]])
        if self.save:
            if not os.path.isdir(d_path):
                os.mkdir(d_path)
            d_path = os.path.join(d_path,str(index).zfill(5))
            np.savez_compressed(d_path,pvh=mcv,label=label,mid = mid,len=leng)
            if index % 100 == 0:
                print '{0} {1} {2}/{3}  {4}/{5} saved'.format(info[0], info[1], index, data['label'].shape[0],item,self.length)
        # print time()-t
        # return {'data':self.frame_data,'label':label}
        return mcv,mid,leng

    def __len__(self):
        return len(self.data)


class Human36V(Dataset):
    """
    Read processed volume data from Human3.6M dataset.
    """
    def __init__(self,root_path):
        self.data_augmentation = False
        self.subjects = filter(lambda x: os.path.isdir(os.path.join(root_path, x)), os.listdir(root_path))
        self.subjects.sort()
        self.subjects = ['S1','S5','S6','S7','S8','S9','S11']
        # self.subjects = ['S1']
        self.length = 0
        self.training_subjects = ['S1','S5','S6','S7','S8']
        # self.training_subjects = ['S1']
        self.test_subjects = ['S9','S11']
        # self.test_subjects = ['S9']
        # self.test_subjects = []

        # encapsulate into class
        self.subjects = [Subject(i) for i in self.subjects]
        self.training_subjects = [Subject(i) for i in self.training_subjects]
        self.test_subjects = [Subject(i) for i in self.test_subjects]

        self.data_dict = {}
        self.data = []
        self.test_lengths = []

        # for i in range(len(self.training_subjects)):
        #     self.training_subjects[i].actionName = self.training_subjects[i].actionName[:1]
        # self.test_subjects[0].actionName = self.test_subjects[0].actionName[:1]
        print 'load training set'
        for sub in self.training_subjects:
            self.data_dict[sub.name] = {}
            for act in sub.actionName:
                print 'load',sub.name,act,
                self.data_dict[sub.name][act] = {}

                p = os.path.join(root_path, sub.name, 'MySegmentsMat', 'ground_truth_bs',act,'*.npz')
                lines = glob.glob(p)
                lines.sort()
                if lines is None or len(lines) == 0:
                    print '0 loaded'
                    continue
                for i in range(len(lines)):
                    self.data.append(lines[i])
                print len(lines),'loaded'
                self.length += len(lines)
        self.training_length = self.length
        # load test set
        print 'load test set'
        for sub in self.test_subjects:
            self.data_dict[sub] = {}
            for act in sub.actionName:
                print 'load', sub, act,
                self.data_dict[sub][act] = {}
                p = os.path.join(root_path, sub.name, 'MySegmentsMat', 'ground_truth_bs',act,'*.npz')
                lines = glob.glob(p)
                lines.sort()
                if lines is None or len(lines) == 0:
                    print '0 loaded'
                    continue
                for i in range(len(lines)):
                    self.data.append(lines[i])
                print len(lines),'loaded'
                self.test_lengths.append(len(lines))
                self.length += len(lines)
        self.test_length = self.length-self.training_length
        # self.subjects_length.append(sub_len)

    def __getitem__(self, item):
        d = self.data[item]
        npz = np.load(d)
        pvh, label, mid, leng = npz['pvh'],npz['label'],npz['mid'],npz['len']
        pvh = pvh.astype(np.uint8)
        # if self.data_augmentation:
        #     pvh = random_cut(pvh)
        pvh = torch.from_numpy(pvh).cuda().float()
        # if self.data_augmentation:
        #     pvh,label,mid,leng = data_augmentation3D(pvh,label,mid,leng)

        return pvh,label,mid.astype(np.float32),leng.reshape((1)).astype(np.float32)

    def __len__(self):
        return self.length

    def get_subset(self,start,end):
        sub = range(int(self.length * start), int(self.length*end))
        return sub

    def get_train_test(self):
        """
        get train test dataset
        :return: train indices, test indices
        """
        train = range(self.training_length)
        test = range(self.training_length,self.length)
        return train,test

    def get_config(self):
        """
        :return: training subjects, test subjects
        """
        s = reduce((lambda x,y:x+y),self.subjects)
        ts = reduce((lambda x,y:x+y),self.test_subjects)
        return s,ts


def get_sample_image(path,index):
    inputparameters = {}
    outputparameters = {}
    outputparameters['-pix_fmt'] = 'gray'
    reader = skvideo.io.FFmpegReader(path,
                                     inputdict=inputparameters,
                                     outputdict=outputparameters)
    i = 0
    for frame in reader.nextFrame():
        # do something with the ndarray frame
        if index == i:
            reader.close()
            return frame
        i+=1


def movefile(s):
    """
    move odd frame to folder 'copy'
    :param s:
    :return:
    """
    import shutil
    p = '/home/hfy/data/human36/{0}/MySegmentsMat/ground_truth_bs'.format(s)
    pto = '/home/hfy/data/human36/{0}/MySegmentsMat/copy'.format(s)
    if not os.path.isdir(pto):
        os.mkdir(pto)
    for l in os.listdir(p):
        dpath = os.path.join(p, l)
        if os.path.isdir(dpath):
            if not os.path.isdir(os.path.join(pto, l)):
                os.mkdir(os.path.join(pto, l))
            fname = os.listdir(dpath)
            fname.sort()
            for i in range(len(fname)):
                if i % 2 != 0:
                    shutil.move(os.path.join(dpath,fname[i]),os.path.join(pto,l,fname[i]))
                    # os.move(os.path.join(dpath, fname[i]))
            print l


def movefile1(s):
    import shutil
    p = '/home/hfy/data/human36/{0}/MySegmentsMat/ground_truth_bs'.format(s)
    pto = '/home/hfy/data/human36/{0}/MySegmentsMat/copy'.format(s)
    for l in os.listdir(pto):
        dpath = os.path.join(pto, l)
        if os.path.isdir(dpath):
            # if not os.path.isdir(os.path.join(pto, l)):
            #     os.mkdir(os.path.join(pto, l))
            fname = os.listdir(dpath)
            fname.sort()
            for i in range(len(fname)):
                shutil.move(os.path.join(pto,l,fname[i]),os.path.join(p,l,fname[i]))
            print l


if __name__ == '__main__':
    ds = Human36(HM_PATH)
    print len(ds)
    ds[0]
    # tc.raw = False
    # tc.save = True
    # for i in range(len(tc)):
    #     tc[i]
    # tc.data_augmentation=True
    # timeit()
    # for i in range(100):
    #     s = tc[i]

        # timeit()
        # visualize_sample(tc[0])
    # for i in range(10):
    #     tc.data_augmentation = False
    #     visualize_sample(tc[i])
    #     tc.data_augmentation = True
    #     visualize_sample(tc[i])

        # f = tc[0]
    # for i in range(1000):
    #     print tc[i]['data'][0].shape
    # import PIL.Image as Image
    # Image.fromarray(f['data']).show()
