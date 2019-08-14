from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
#from dataloader import KITTIloader2015 as ls
from models import *
from models.submodule import scale_pyramid

parser = argparse.ArgumentParser(description='TNET')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--max_disp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/home/jianing/hdd/sceneflow/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set gpu id used
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# if args.KITTI == '2015':
#     from dataloader import KITTIloader2015 as ls
# else:
#     from dataloader import KITTIloader2012 as ls  

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left, all_right, all_gt, test_left, test_right, test_gt = lt.dataloader(args.datapath)

Trainleftoader = torch.utils.data.DataLoader(
            DA.myImageFloder(all_left,all_right,all_gt, True), 
            batch_size= 12, shuffle= True, num_workers= 8, drop_last=False)

Testleftoader = torch.utils.data.DataLoader(
            DA.myImageFloder(all_left,all_right,all_gt, False), 
            batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)


# if args.model == 'stackhourglass':
#     model = stackhourglass(args.maxdisp)
# elif args.model == 'basic':
#     model = basic(args.maxdisp)
# else:
#     print('no model')
model = wnet(args.max_disp)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

def train(left,right,gt):
        model.train()
        left   = Variable(torch.FloatTensor(left))
        right   = Variable(torch.FloatTensor(right))   
        gt = Variable(torch.FloatTensor(gt))

        if args.cuda:
            left, right, gt = left.cuda(), right.cuda(), gt.cuda()

        #---------
        
        #gt_pyramid = scale_pyramid(gt)
        #mask = [g < args.max_disp for g in gt_pyramid]
        #mask = [m.float().detach_() for m in mask]
        mask = gt < args.max_disp
        mask = mask.float()
        mask.detach_()
        #----
        optimizer.zero_grad()

        output = model(left,right)
        output = [torch.squeeze(o,1) for o in output]
        weight = [1,1,1,0.8]
        loss = [weight[i]*F.smooth_l1_loss(output[i]*mask, gt*mask, size_average=True) for i in range(4)]
        print("first loss %f, second loss %f, third loss %f, fourth loss %f"%(loss[0],loss[1],loss[2],loss[3]))
        loss = sum(loss)

        loss.backward()
        optimizer.step()

        return loss.data

def test(left,right,gt):
        model.eval()
        left   = Variable(torch.FloatTensor(left))
        right   = Variable(torch.FloatTensor(right))   
        if args.cuda:
            left, right = left.cuda(), right.cuda()

        #---------
        mask = gt < 192
        #----

        with torch.no_grad():
            output3 = model(left,right)

        output = torch.squeeze(output3.data.cpu(),1)[:,4:,:]

        if len(gt[mask])==0:
            loss = 0
        else:
            loss = torch.mean(torch.abs(output[mask]-gt[mask]))  # end-point-error

        return loss

def adjust_learning_rate(optimizer, epoch):
    lr = 1e-3
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    start_full_time = time.time()
    for epoch in range(1, args.epochs+1):
        print('This is %d-th epoch'%(epoch))
        total_test_loss = 0
        adjust_learning_rate(optimizer,epoch)

        for batch_idx,(left,right,gt) in enumerate(Trainleftoader):
            start_time = time.time()
            loss = train(left,right,gt)
            print('Iter %d training loss = %.3f,time=%.2f'%(batch_idx,loss,time.time()-start_time))
    print('full training time = %.2f hr'%((time.time() - start_full_time)/3600))

if __name__ == '__main__':
    main()
    
