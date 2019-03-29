from torch.utils.data import Dataset,DataLoader
import numpy as np
import os,glob
import skvideo.io
from params import *
from server_setting import *
from mcv import make_rgb_mcv
from camera_hm import *
import glob
from time import time

t = 0
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
        self._rgbpath = os.path.join(HM_RGB_PATH, name, 'Videos')
        self.import_data()

    def import_data(self):
        """
        init subject and action
        :return:
        """
        # matte videos
        p = os.path.join(HM_PATH,self.name, 'MySegmentsMat', 'ground_truth_bs','*.mp4')
        matte_videos = glob.glob(p)
        for file in matte_videos:
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
       # print self.pvhName
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
        #print self.actionName
        #print self.actionDict[self.actionName[0]] # ['Directions.csv', 'Directions.54138969.mp4', 'Directions.55011271.mp4', 'Directions.58860488.mp4', 'Directions.60457274.mp4']

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
        #print gt
        # with h5py.File(gtfile, 'r') as hf:
        #     points = hf[hf.keys()[0]][()]
        # return [os.path.join(self._bspath,x) for x in self.actionDict[actionName][1:]], points.T
        return [os.path.join(self._bspath,x) for x in self.actionDict[actionName][1:]], \
               [os.path.join(self._rgbpath, x) for x in self.actionDict[actionName][1:]],\
               gt

    def __str__(self):
        return self.name


class Human36RGB(Dataset):
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
        self.subjects = ['S1','S5','S6','S7','S8','S9','S11']
      #  self.training_subjects = ['S1','S5','S6','S7','S8']
      #  self.test_subjects = ['S9','S11']
        self.training_subjects = ['S1']
        self.test_subjects = ['S9']
        #self.subjects = ['S1']
        self.subjects = [Subject(i) for i in self.subjects]
        self.training_subjects = [Subject(i) for i in self.training_subjects]
        self.test_subjects = [Subject(i) for i in self.test_subjects]

        self.data_dict = {}
        self.data = []
        self.length = 0
        self.matte_video_readers = [MyReader() for i in range(NUM_CAM_HM)]
        self.rgb_video_readers = [MyReader() for i in range(NUM_CAM_HM)]
        self.frame_data = [None]*NUM_CAM_HM*2
        self.save = False
        self.raw = False
 #       self.test_lengths = [] #list for each subject and action in testset
        # self.subjects[0].actionName = ['Directions']
        self.cams = load_cameras(os.path.join(HM_PATH,'cameras.h5'))
        print 'load training set :'
        for sub in self.training_subjects:
            self.data_dict[sub.name] = {}
            # self.data_dict[sub]['camera'] = self.cams[sub]
            for act in sub.actionName:
                self.data_dict[sub.name][act] = {}
                print 'load',sub,act  # load S1 Directions + kpts location
                matte_videos,rgb_videos, label = sub.select_action(act) # four matte & rgb videos once ,__len__ = 4;;label shape= <type'tuple'>:(1612,96)
                lines = label.shape[0] # eg.1612
                self.data_dict[sub.name][act]['label'] = label
                self.data_dict[sub.name][act]['matte_video'] = matte_videos
                self.data_dict[sub.name][act]['rgb_video'] = rgb_videos
                for i in range(lines):
                    self.data.append([sub.name, act, i])
                #print lines, 'loaded'
                self.length += lines
        self.training_length = self.length
        print 'training length is :', self.training_length
        # load test set
        print 'load test set :'
        for sub in self.test_subjects:
            self.data_dict[sub.name] = {}
            # self.data_dict[sub]['camera'] = self.cams[sub]
            for act in sub.actionName:
                self.data_dict[sub.name][act] = {}
                print 'load',sub,act  # load S1 Directions + kpts location
                matte_videos,rgb_videos, label = sub.select_action(act) # four matte & rgb videos once ,__len__ = 4;;label shape= <type'tuple'>:(1612,96)
                lines = label.shape[0] # eg.1612
                self.data_dict[sub.name][act]['label'] = label
                self.data_dict[sub.name][act]['matte_video'] = matte_videos
                self.data_dict[sub.name][act]['rgb_video'] = rgb_videos
                for i in range(lines):
                    self.data.append([sub.name, act, i])
               # print lines, 'loaded'

               # self.test_lengths.append(lines)
                self.length += lines
        self.test_length = self.length-self.training_length
        print 'test length is :',self.test_length
        print 'total length is: ',self.length

        # load camera params
        # self.subjects_length.append(sub_len)

    def __getitem__(self, item): #eg. in S1,item is from 0 to 62094 frames
        """
        get specific frame
        :param item: index of frame
        :return:
        """
        global t
        info = self.data[item] #type(info)=list:['S1','Directions',item]
        # info = [sub.name, act, i]
        data = self.data_dict[info[0]][info[1]]
        index = info[2]
        matte_videos = data['matte_video']
        rgb_videos = data['rgb_video']
        label = data['label'][index] # [index,96] the index/th row info,(96,)
        d_path = os.path.join(os.sep, os.path.join(*rgb_videos[0].split(os.sep)[1:-1]),info[1])
        np_path = os.path.join(d_path, str(index).zfill(5)+'.npz')
        if not self.raw and os.path.isfile(np_path):
            npz = np.load(np_path)
            mcv, label, mid, leng = npz['mcv'], npz['label'], npz['mid'], npz['len'] #mcv:,label:(96,),mid:(3,),len:(1,)
            mcv = mcv.astype(np.uint8)
            mcv = torch.from_numpy(mcv).cuda().float()
            return mcv, label.astype(np.float32), mid.astype(np.float32), leng.reshape((1)).astype(np.float32)
        for i in range(NUM_CAM_HM):
            # current video reader not matched
            if self.matte_video_readers[i].current_video != matte_videos[i]:
                self.matte_video_readers[i].current_video = matte_videos[i]
                self.rgb_video_readers[i].current_video = rgb_videos[i]

                # set keys and values for parameters in ffmpeg
                inputparameters = {}
                outputparameters = {}
                outputparameters['-pix_fmt'] = 'gray'
                rgb_outputparameters = {}
                if self.matte_video_readers[i].readers is skvideo.io.FFmpegReader:
                    self.matte_video_readers[i].readers.close()
                if self.rgb_video_readers[i].readers is skvideo.io.FFmpegReader:
                    self.rgb_video_readers[i].readers.close()

                if not os.path.isfile(self.rgb_video_readers[i].current_video):
                    print("file not found",self.rgb_video_readers[i].current_video)
                    return None
                self.matte_video_readers[i].readers = skvideo.io.FFmpegReader(self.matte_video_readers[i].current_video,
                                                 inputdict=inputparameters,
                                                 outputdict=outputparameters)

                self.rgb_video_readers[i].readers = skvideo.io.FFmpegReader(self.rgb_video_readers[i].current_video,
                                                                              inputdict=inputparameters,
                                                                              outputdict=rgb_outputparameters)
                print 'load',self.matte_video_readers[i].current_video,self.matte_video_readers[i].readers.inputframenum,'frames'
                print 'load rgb',self.rgb_video_readers[i].current_video,self.rgb_video_readers[i].readers.inputframenum,'frames'
                if index<0 or index >= self.matte_video_readers[i].readers.inputframenum or index >= self.rgb_video_readers[i].readers.inputframenum:
                    print 'frame overloaded ',matte_videos[0], index, self.matte_video_readers[i].readers.inputframenum
                    return None
                    # raise ValueError("index not valid")
                self.matte_video_readers[i].current_frame = 0
                self.rgb_video_readers[i].current_frame = 0
                for frame in self.matte_video_readers[i].readers.nextFrame():
                    # determine frame number
                    if self.matte_video_readers[i].current_frame == index:
                        self.matte_video_readers[i].current_frame+=1
                        self.frame_data[i] = frame
                        break
                    else:
                        self.matte_video_readers[i].current_frame+=1

                for frame in self.rgb_video_readers[i].readers.nextFrame():
                    # determine frame number
                    if self.rgb_video_readers[i].current_frame == index:
                        self.rgb_video_readers[i].current_frame += 1
                        self.frame_data[i+NUM_CAM_HM] = frame
                        break
                    else:
                        self.matte_video_readers[i].current_frame += 1
            # video matched
            else:
                if self.matte_video_readers[i].readers is None or \
                   self.rgb_video_readers[i].readers is None or \
                   index < self.matte_video_readers[i].current_frame or \
                   index >= self.matte_video_readers[i].readers.inputframenum or \
                   index >= self.rgb_video_readers[i].readers.inputframenum:
                    # print matte_videos[0],index,self.matte_video_readers[i].current_frame
                    return None
                    # raise ValueError("index not valid")
                for frame in self.matte_video_readers[i].readers.nextFrame():
                    # determine frame number
                    if self.matte_video_readers[i].current_frame == index:
                        self.matte_video_readers[i].current_frame += 1
                        self.frame_data[i] = frame
                        break
                    else:
                        self.matte_video_readers[i].current_frame += 1

                for frame in self.rgb_video_readers[i].readers.nextFrame():
                    # determine frame number
                    if self.rgb_video_readers[i].current_frame == index:
                        self.rgb_video_readers[i].current_frame += 1
                        self.frame_data[i+NUM_CAM_HM] = frame
                        break
                    else:
                        self.rgb_video_readers[i].current_frame += 1

        # make mcv
        # t = time
        if self.raw:
            return self.frame_data,label
        mcv,mid,leng = make_rgb_mcv(self.frame_data, label, self.cams[info[0]])
        if self.save:
            if not os.path.isdir(d_path):
                os.mkdir(d_path)
            d_path = os.path.join(d_path,str(index).zfill(5))
            np.savez_compressed(d_path,mcv=mcv,label=label,mid = mid,len=leng)
            if index % 100 == 0:
                print '{0} {1} {2}/{3}  {4}/{5} saved in {6}s'.format(info[0], info[1], index, data['label'].shape[0],item,self.length,(time()-t))
                t = time()

            return 0
        # print time()-t
        # return {'data':self.frame_data,'label':label}
        return mcv,mid,leng

    def __len__(self):
        return len(self.data)

    def get_train_test(self):
        """
        get train test dataset
        :return: train indices, test indices
        """
        train = range(self.training_length)
        test = range(self.training_length, self.length)
        return train, test


class Human36RGBV(Dataset):
    """
    Read processed volume data from Human3.6M dataset.
    """
    def __init__(self,root_path):
        self.data_augmentation = False
        self.subjects = filter(lambda x: os.path.isdir(os.path.join(root_path, x)), os.listdir(root_path))
        self.subjects.sort()
        # self.subjects = ['S1','S5','S6','S7','S8','S9','S11']
        self.subjects = ['S1','S9']
        self.length = 0
        #self.training_subjects = ['S1','S5','S6','S7','S8']
        self.training_subjects = ['S1']
        #self.test_subjects = ['S9','S11']
        self.test_subjects = ['S9']
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

                p = os.path.join(root_path, sub.name,'Videos',act,'*.npz')
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
                p = os.path.join(root_path, sub.name,'Videos',act,'*.npz')
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
        pvh, label, mid, leng = npz['mcv'],npz['label'],npz['mid'],npz['len']
        pvh = pvh.astype(np.float32)
        if nCHANNEL == 16:
            pvh[0:3, :, :, :] = pvh[0:3, :, :, :] / 255
            pvh[4:7, :, :, :] = pvh[4:7, :, :, :] / 255
            pvh[8:11, :, :, :] = pvh[8:11, :, :, :] / 255
            pvh[12:15, :, :, :] = pvh[12:15, :, :, :] / 255
        elif nCHANNEL == 12:
            pvh[0:3, :, :, :] = pvh[0:3, :, :, :] / 255
            pvh[3:6, :, :, :] = pvh[4:7, :, :, :] / 255
            pvh[6:9, :, :, :] = pvh[8:11, :, :, :] / 255
            pvh[9:12, :, :, :] = pvh[12:15, :, :, :] / 255
            pvh = pvh[0:12,:,:,:]

        # if self.data_augmentation:
        #     pvh = random_cut(pvh)
        pvh = torch.from_numpy(pvh).cuda()
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




def preprocess():
    t = time()
    h = Human36RGB(HM_PATH)
    # h.raw = True
    h.save = True
    for i in range(len(h)):
        r = h[i]
        if r == 0:
            continue
        if i % 100 == 0:
            print '{0}/{1} processed'.format(i,len(h))


def show_raw():
    import cv2
    h = Human36RGB(HM_PATH)
    h.raw = True
    for i in range(len(h)):
        r = h[i]
        for j in range(4):
            cv2.imshow('',r[0][j])
            cv2.waitKey(0)
            cv2.imshow('',r[0][j+NUM_CAM_HM])
            cv2.waitKey(0)


def check_volume():
    from visualization import plot_voxel_label, plot_voxel
    ds = Human36RGB(HM_PATH)
    ds.save = False
    for i in range(0, 1):
        data, label, mid, leng = ds[i]
        # data = data.cpu()
        label = torch.from_numpy(label)
        mid = torch.from_numpy(mid)
        leng = torch.from_numpy(leng)
        print data.shape, label.shape, mid.shape, leng.shape #(16,64,64,64)(96,) (3,)(1,)
        # s = data.sum(dim=0)
        # s[s < 4] = 0
        for j in data: # j.shape :(64,64,64)
            plot_voxel(j)
        # leng = leng * (NUM_VOXEL / NUM_GT_SIZE)
        # base = mid - leng.repeat(1, 3) * (NUM_GT_SIZE / 2 - 0.5)
        # leng = leng.repeat(1, JOINT_LEN * 3)
        # base = base.repeat(1, JOINT_LEN)
        # plot_voxel_label(s, (label - base) / (leng / (NUM_VOXEL / NUM_GT_SIZE)))



if __name__ == '__main__':
    ds = Human36RGB(HM_PATH)
    print len(ds)
    ds[0]
# preprocess()
# show_raw()
#check_volume()