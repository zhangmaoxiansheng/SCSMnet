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
from models.submodule import decompose_disp_pro
from models.submodule import sparse_test
from models.submodule_DSR import DSR_loss
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='TNET')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--max_disp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--max_range', type=int ,default=32,
                    help='maxium disparity')
parser.add_argument('--learning_rate', type=float ,default=1e-3,
                    help='lr')
parser.add_argument('--mask_disp', type=float ,default=1,
                    help='define the mask range')
parser.add_argument('--stage', default='forward',
                    help='training stage')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/home/jianing/hdd/sceneflow/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./weight',
                    help='save model')
parser.add_argument('--log_dir', default='/home/jianing/hdd/runs',
                    help='logdir')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
if not os.path.exists(args.savemodel):
    os.mkdir(args.savemodel)
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

Trainloader = torch.utils.data.DataLoader(
            DA.myImageFloder(all_left,all_right,all_gt, True), 
            batch_size= 12, shuffle= True, num_workers= 8, drop_last=False)

Testloader = torch.utils.data.DataLoader(
            DA.myImageFloder(all_left,all_right,all_gt, False), 
            batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)


# if args.model == 'stackhourglass':
#     model = stackhourglass(args.maxdisp)
# elif args.model == 'basic':
#     model = basic(args.maxdisp)
# else:
#     print('no model')
model = wnet(args.max_disp,args.stage,args.max_range)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    if args.stage == 'distill':
        model_dict = model.state_dict()
        pretrained_dict_ = torch.load(args.loadmodel)
        pretrained_dict_ = pretrained_dict_['state_dict']
        #for k,v in pretrained_dict_.items():
            #print(k)
        pretrained_dict = {k:v for k,v in pretrained_dict_.items() if k in model_dict}
        model_dict.update(pretrained_dict)#items and update are dict function
        model.load_state_dict(model_dict)
        for k,v in model.named_parameters():
            if k in pretrained_dict_:
                v.requires_grad = False
            if 'DSR' in k:
                v.requires_grad = True
    else:
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])


print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

def P1_loss(vol_test,vol_gt,mask):
    B,D,H,W = vol_test.size()
    loss = 0
    mask = F.interpolate(mask.unsqueeze(1),[H,W]).squeeze(1)
    for i in range(D):
        vol_test_layer = vol_test[:,i,:,:]
        vol_gt_layer = vol_gt[:,i,:,:]
        loss += F.smooth_l1_loss(vol_test_layer * mask,vol_gt_layer * mask,size_average=True)
    return loss/D
def generate_mask(res_gt, disp_range,mask_pyramid):
    mask = [torch.abs(res_gt[i]) > args.mask_disp for i in range(4)]
    mask = [m.float() for m in mask]
    mask = [mask[i] * mask_pyramid[i] for i in range(4)]
    mask = [m.detach_() for m in mask]
    return mask
def generate_output(res_output, base):
    res_output = [torch.squeeze(d,1) for d in res_output if d is not None]
    output = [res_output[i] + base[i] for i in range(4)]
    return res_output, output

def train(left,right,gt,epochs):
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
        if args.stage == 'forward':
            volume_16 = decompose_disp_pro(gt*mask,16,args.max_disp//16)
            volume_8 = decompose_disp_pro(gt*mask,8,args.max_disp//8)
            volume_4 = decompose_disp_pro(gt*mask,4,args.max_disp//4)
            volumes = [volume_4,volume_8,volume_16]
            output = [torch.squeeze(o) for o in output]
            weight = [1,0.5,0.25]
            loss = [F.smooth_l1_loss(output[i]*mask, gt*mask, size_average=True) for i in range(3)]
            #res_loss = 5*F.smooth_l1_loss(output[4]*mask,(gt-output[1])*mask)
            #diff_loss = 0.5*torch.exp(loss[0]-loss[1])
            volume_loss = [P1_loss(output[i],volumes[i-3],mask) for i in range(3,6)]
        
            #print('training volume sparse %f'%sparse_test(output[3]))

            loss_ = [loss[i]*weight[i] for i in range(3)]
            volume_loss = [volume_loss[i]*weight[i] for i in range(3)]
            volume_loss = sum(volume_loss)
            print("first loss %f volume loss %f"%(loss[0],volume_loss))
            loss_f = sum(loss_)+volume_loss*5
        else:
            
            disp_gt = scale_pyramid(gt.unsqueeze(1))
            disp_gt = [torch.squeeze(d,1) for d in disp_gt]
            lr_pyramid = output[4]
            mask_pyramid = scale_pyramid(mask.unsqueeze(1))
            mask_pyramid = [torch.squeeze(m,1) for m in mask_pyramid]

            res_gt = [disp_gt[i] - lr_pyramid[i]*mask_pyramid[i] for i in range(4)]
            mask_ = generate_mask(res_gt, args.mask_disp,mask_pyramid)
            res_output = output[:4]
            res_output, output = generate_output(res_output, lr_pyramid)
            
            L = DSR_loss(res_output, output, lr_pyramid, disp_gt, mask_,mask_pyramid)
            #total_loss = 1.5 * l1_res_loss + 1 * SSIM_loss + 2.5 * diff_loss + 0.7 * lr_l1_loss
            loss_f = 1.5 * L.l1_res_loss + 1 * L.SSIM_loss + 4 * L.diff_loss + 0.2 * L.lr_l1_loss + 1.5 * (L.l1_full_loss + L.SSIM_full_loss)
            l1_loss = L.l1_res_loss
            o_loss_ = L.o_loss
            progress = o_loss_ - l1_loss
            loss = L.true_l1
            final_output = output[0]
            print("first loss %f oigin_loss %f progress %f"%(loss,o_loss_,progress))

        loss_f.backward()
        optimizer.step()

        return loss_f.data, loss

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
        output = output3[0]
        output = torch.squeeze(output.data.cpu(),1)[:,4:,:]

        if len(gt[mask])==0:
            loss = 0
        else:
            loss = torch.mean(torch.abs(output[mask]-gt[mask]))  # end-point-error

        return loss

def adjust_learning_rate(optimizer, epoch,init_lr):
    lr = init_lr
    if(epoch>7):
        lr = lr/10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    start_full_time = time.time()
    writer = SummaryWriter(log_dir=str(args.log_dir))
    global_step = 0
    for epoch in range(1, args.epochs+1):
        print('This is %d-th epoch'%(epoch))
        total_test_loss = 0
        adjust_learning_rate(optimizer,epoch,args.learning_rate)

        for batch_idx,(left,right,gt) in enumerate(Trainloader):
            start_time = time.time()
            loss_total,loss_ = train(left,right,gt,epoch)
            global_step +=1
            print('Epoch %d Iter %d training loss = %.3f,time=%.2f'%(epoch,batch_idx,loss_total,time.time()-start_time))
            if batch_idx % 5 == 0:
                writer.add_scalar('loss/total_loss', loss_total, global_step=global_step)
                writer.add_scalar('loss/loss1', loss_[0], global_step=global_step)
            if((global_step+1)%2000 == 0):
                savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'_'+str(batch_idx)+'.tar'
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),    
                }, savefilename)
    print('full training time = %.2f hr'%((time.time() - start_full_time)/3600))
    writer.close()
    savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'_'+str(batch_idx)+'.tar'
    torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),    
                }, savefilename)   
    #------------- TEST ------------------------------------------------------------
    # total_test_loss = 0
    # for batch_idx, (imgL, imgR, disp_L) in enumerate(Testloader):
    #     test_loss = test(imgL,imgR, disp_L)
    #     print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
    #     total_test_loss += test_loss

    # print('total test loss = %.3f' %(total_test_loss/len(Testloader)))
	# #----------------------------------------------------------------------------------
	# #SAVE test information
    # savefilename = args.savemodel+'testinformation.tar'
    # torch.save({
    #         'test_loss': total_test_loss/len(Testloader),
    #     }, savefilename)

if __name__ == '__main__':
    main()
    
