import argparse
import datetime
import shutil
import warnings
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.human36rgb import *
from params import *
from server_setting import *
from time import time
from model.FUSENet import FuseNet
from helper import AverageMeter,timeit
from metrics import mean_error
# from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from subset_sampler import SubsetSampler
from parallel import DataParallelModel, DataParallelCriterion

# init setting
def init_parser():
    parser = argparse.ArgumentParser(description='Fusion')
    parser.add_argument('-data',default=HM_PATH, type=str, metavar='DIR',
                        help='path to dataset(default: {})'.format(HM_PATH))

    parser.add_argument('-e', '--epochs',  default=EPOCH_COUNT, type=int,
                        help='number of total epochs to run (default: {})'.format(EPOCH_COUNT))

    parser.add_argument('-s', '--start-epoch',  default=0, type=int,
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                         help='mini-batch size (default: {})'.format(BATCH_SIZE))

    parser.add_argument('-lr', '--learning-rate', default=LEARNING_RATE, type=float,
                        metavar='LR', help='initial learning rate (default: {})'.format(LEARNING_RATE))

    parser.add_argument('-m', '--momentum', default=MOMENTUM, type=float, metavar='M',
                        help='momentum (default: {})'.format(MOMENTUM))

    parser.add_argument('-wd', '--weight-decay', default=WEIGHT_DECAY, type=float,
                        metavar='W', help='weight decay (default: {})'.format(WEIGHT_DECAY))

    parser.add_argument('-p', '--print-freq', default=PRINT_FREQ, type=int,
                         help='print frequency (default: {})'.format(PRINT_FREQ))

    parser.add_argument('-g', '--gpu-id', default=GPU_ID, type=int,
                         help='GPU ID (default: {})'.format(GPU_ID))

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('-n','--name', default='pth', type=str, metavar='PATH',
                        help='result name')

    global args
    args = parser.parse_args()


def warning_init():
    warnings.filterwarnings("error")
    np.seterr(all='warn')


# main body
def train_human(full = False):
    Fuse = FuseNet(nSTACK, nModule, nFEAT, JOINT_LEN)
  #  net = FuseNet(nSTACK, nModule, nFEAT, JOINT_LEN)
  #   net = nn.DataParallel(Fuse,device_ids=[0])
    net = Fuse.cuda()
    warning_init()
    start_time = time()
 #   net.cuda()
  #  net = nn.DataParallel(net)
  #   net.to(device)
    #net = DataParallelModel(net)
    criterion = nn.MSELoss().cuda()
    # criterion = DataParallelCriterion(criterion)
    best_err = 99990
    optimizer_rms = optim.RMSprop(net.parameters(),
                                  lr=args.learning_rate,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)

    optimizer_sgd = optim.SGD(net.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    optimizer = optimizer_rms
    # optimizer = torch.nn.DataParallel(optimizer_rms,device_ids=[0])
    # resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #dataset = Human36V(HM_PATH)
    dataset = Human36RGBV(HM_RGB_PATH)
    dataset.data_augmentation = True

    train_idx, valid_idx = dataset.get_train_test()
    # train_idx = range(2)
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetSampler(valid_idx)


    if full:
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=WORKER,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=TEST_BATCH,
                                                  num_workers=WORKER,
                                                  shuffle=False)

    else:
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size, sampler=train_sampler,
                                                   num_workers=WORKER)

        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=TEST_BATCH, sampler=test_sampler,
                                                  num_workers=WORKER)
    optimizer = optimizer_rms
    set_learning_rate(optimizer, args.learning_rate)

    for epoch in range(args.start_epoch, args.epochs):  # loop over the dataset multiple times
        print ('best error:', best_err)
        epoch_start_time = time()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        adjust_learning_rate(optimizer, epoch)

        # train set
        dataset.data_augmentation = True
        loss, err = train(train_loader, net, criterion, optimizer, epoch + 1)
        print 'training error is ', err

        # test set
        dataset.data_augmentation = False
        err = test(test_loader, net, criterion)[2]
        print 'test error is', err

        # remember best acc and save checkpoint
        is_best = err < best_err
        best_err = min(err, best_err)

        print('Epoch: [{0}/{1}]  Time [{2}/{3}]'.format(
            epoch + 1, args.epochs, datetime.timedelta(seconds=(time() - epoch_start_time)),
            datetime.timedelta(seconds=(time() - start_time))))

        path = "checkpoint/checkpoint_{0}_{1}.{2}.tar".format(epoch, get_learning_rate(optimizer), args.name)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': '3dHMP',
            'state_dict': net.state_dict(),
            'best_acc': best_err,
            'optimizer': optimizer.state_dict(),
        }, is_best, 'checkpoint/checkpoint_{0}.{1}.tar'.format(epoch,args.name))

    '''    if epoch % DECAY_EPOCH == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': '3dHMP',
                'state_dict': net.state_dict(),
                'best_acc': best_err,
                'optimizer': optimizer.state_dict(),
            }, False, path)  '''
            # if not is_best:
            #     lower_learning_rate(optimizer,DECAY_RATIO)

    print('Finished Training')


def test_human(path):
    start_time = time()
    save = True
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    fusenet = FuseNet(nSTACK, nModule, nFEAT,JOINT_LEN_HM)
    fusenet.cuda()
    fusenet.load_state_dict(checkpoint['state_dict'])
    fusenet.eval()

    dataset = Human36RGBV(HM_RGB_PATH)
    dataset.data_augmentation = False
    criterion = nn.MSELoss().cuda()
  #  criterion = DataParallelCriterion(criterion)
    best_acc = checkpoint['best_acc']
    print ("using model with acc [{:.2f}]".format(best_acc))

    train_idx, valid_idx = dataset.get_train_test()
    train_sampler = SubsetSampler(train_idx)
    test_sampler = SubsetSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1, sampler=train_sampler,
                                               num_workers=WORKER)

    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=TEST_BATCH, sampler=test_sampler,
                                              num_workers=WORKER)

    r, l, acc = test(test_loader, fusenet, criterion)

    print("final accuracy {:3f}".format(acc))
    print("total time: ", datetime.timedelta(seconds=(time() - start_time)).seconds, 's')
    if save is True:
        np.savez_compressed('result/human_p2_c12.npz',result=r,label=l)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    # switch to train mode
    model.train()

    end = time()
    for i, s in enumerate(train_loader):
        data, label, mid, leng= s
        #c1,c2,c3,c4 = data.cpu()

        # measure data loading time
        data_time.update(time() - end)
        batch_size = data.size(0)
        # input_var = torch.autograd.Variable(data.to(device).float())
        input_var = torch.autograd.Variable(data)
        # target_var = torch.autograd.Variable(label.to(device))
        target_var = torch.autograd.Variable(label.cuda())
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs")
        #     output = model(input_var)
        # else:
            # print("Only 1 GPU found")
        output = model(input_var)

        # record loss
        # leng is voxel length
        leng = leng*(NUM_VOXEL/NUM_GT_SIZE)
        mid = mid - leng.repeat(1, 3) * (NUM_GT_SIZE / 2 - 0.5)
        leng = leng.repeat(1, JOINT_LEN * 3)
        base = mid.repeat(1, JOINT_LEN)

        for j in range(len(output)):
            output[j] = (output[j].mul(leng.cuda())).add(base.cuda())
        loss = criterion(output[0], target_var)
        for k in range(1, nSTACK):
            loss += criterion(output[k], target_var)
            # print loss
        losses.update(loss.item()/batch_size, 1)
        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        err_t = float(mean_error(output[-1].cpu(),label)[0])
        # print err_t
        errors.update(err_t, batch_size)


        # measure elapsed time
        batch_time.update(time() - end)
        end = time()

        if i % args.print_freq == 0:
            s_ep = 'Epoch: [{0}][{1}/{2}]'.format(epoch,i, len(train_loader))
            s_acc = 'Acc {err_t.val:.2f} ({err_t.avg:.2f})'.format(err_t=errors)
            s_loss = 'Loss {loss.val:.2f} ({loss.avg:.2f})'.format(loss=losses)
            s_time = 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time)
            s_data = 'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(data_time=data_time)
            print('{:28}{:25}{:27}{:20}{}'
                  .format(s_ep, s_acc, s_loss, s_time, s_data))
    return losses.avg, errors.avg


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time()
    result = np.empty(shape=(0,JOINT_POS_LEN),dtype=np.float32)
    label_full = np.empty(shape=(0,JOINT_POS_LEN),dtype=np.float32)

    for i,s in enumerate(test_loader):
        # measure data loading time
        data, label, mid, leng= s
        batch_size = data.size(0)
        input_var = data.cuda().float()
        target_var = label.cuda()
        # forward net
        output = model(input_var)
        # record loss
        leng = leng.cuda()*(NUM_VOXEL/NUM_GT_SIZE)
        base = mid.cuda()-leng.repeat(1,3)*(NUM_GT_SIZE/2-0.5)
        leng = leng.repeat(1, JOINT_LEN * 3)
        base = base.repeat(1,JOINT_LEN)
        for j in range(len(output)):
            output[j] = (output[j].mul(leng)).add(base)
        loss = criterion(output[0], target_var)
        for k in range(1, nSTACK):
            loss += criterion(output[k], target_var)
        losses.update(loss.item()/batch_size, 1)
        output = output[-1].cpu().detach()

        # measure accuracy
        r = mean_error(output, label)
        err_t = float(r[0])
        errors.update(err_t, batch_size)

        # measure elapsed time
        batch_time.update(time() - end)
        end = time()

        if i % TEST_PRINT == 0:
            s_ep = '[{0}/{1}]'.format(i,len(test_loader))
            s_acc = 'Acc {err_t.val:.3f} ({err_t.avg:.3f})'.format(err_t = errors)
            s_loss = 'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(loss = losses)
            s_time = 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time)
            print('{:15}{:25}{:27}{}'
                  .format(s_ep,s_acc,s_loss,s_time))

        # append result to numpy array for saving
        result = np.append(result, output.numpy(), axis=0)
        label_full= np.append(label_full, label.numpy(), axis=0)

    return result,label_full, errors.avg


# helper function
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR"""
    lr = args.learning_rate * (DECAY_RATIO ** (epoch // DECAY_EPOCH))
    print ('adjust Learning rate :',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_learning_rate(optimizer, lr):
    print('Set learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lower_learning_rate(optimizer, ratio):
    """Sets the learning rate to the initial LR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= ratio
    print ('acc up, adjust learning rate to ', param_group['lr'])


def get_learning_rate(optimizer):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print 'result saved to',filename
    if is_best:
        print 'best model got'
        shutil.copyfile(filename, 'model_best.'+args.name+'.tar')


# pre process multi-channel volume data from video
def preprocess():
    data = Human36RGB(HM_PATH)
    data.save = True
    for i in range(len(data)):
        d = data[i]
        if i % 1000 == 0:
            print '{0}/{1} processed'.format(i,len(data))


# visualization data
def check_raw(index):
    from visualization import drawcirclecv
    ds = Human36RGB(HM_PATH)
    ds.save = False
    ds.raw = True

    import cv2
    for i in index:
        frames, label = ds[i]
        label = label.reshape(-1,3)
        # cv2.imshow('k',frames[])
        # print frames[0].shape
        # print label
        for j in range(len(frames)):
            gt = ds.cams['S1'][j].world2pix(torch.from_numpy(label).float().cuda())
            # print gt
            fc = frames[j].copy()
            drawcirclecv(fc, gt)
            cv2.imshow('k',fc)
            # cv2.imwrite(str(j)+'.jpg',fc)
            print i,j
            cv2.waitKey(0)


def check_volume():
    from visualization import plot_voxel_label,plot_voxel
    ds = Human36RGB(HM_PATH)
    ds.data_augmentation = False
    for i in range(0,1):
        data, label, mid, leng= ds[i]
        data = data.cpu()
        label = torch.from_numpy(label)
        mid = torch.from_numpy(mid)
        leng = torch.from_numpy(leng)
        print data.shape,label.shape,mid.shape,leng.shape
        s = data.sum(dim=0)
        s[s<4] = 0
        # for j in data:
        #     plot_voxel(j)
        leng = leng * (NUM_VOXEL / NUM_GT_SIZE)
        base = mid - leng.repeat(1, 3) * (NUM_GT_SIZE / 2 - 0.5)
        leng = leng.repeat(1, JOINT_LEN * 3)
        base = base.repeat(1, JOINT_LEN)
        plot_voxel_label(s, (label - base) / (leng / (NUM_VOXEL / NUM_GT_SIZE)))

if __name__ == "__main__":
    init_parser()
    np.set_printoptions(precision=3,suppress=True)
    #torch.cuda.set_device(args.gpu_id)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    #device_ids = [0, 1, 2, 3]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #drawcirclecv
    # generate mcv
    # preprocess()

    # train and test human3.6
    train_human()

    # test only and save result
    #test_human('/home/alzeng/remote/fyhuang/alzeng/new_deepfuse/deepfuse/model_best.p2_c12_5e.tar')

    # check volume in 3D
    # check_volume()

    # check original 2d image
    # check_raw(range(210,221))
