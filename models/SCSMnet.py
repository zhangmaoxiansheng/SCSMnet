from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
#import sparseconvnet as scn
#################
#feature5 = 2048 H/64 too deep and useless
#feature4 = 1024 H/32
#feature3 = 512 H/16
#feature2 = 256 H/8
#feature1 = 64 H/4
#feature0 = 64 H/2
#################

class SCSMnet(nn.Module):
    
    def __init__(self, max_disp):
        super(SCSMnet, self).__init__()
        self.max_disp = max_disp
        #self.train_stage = train_stage
        layer = [2,3,3,2]
        self.feature_extraction = feature_extraction(layer)
        self.decoder_16 = decoderBlock_dense(5,32,16)
        self.decoder_8 = decoderBlock_dense(5,32+8,16)
        self.decoder_4 = decoderBlock_dense(5,32+8,16,up=False)
        self.decoder_2 = decoderBlock_dense(3,32,16,up=False)
        self.disp_reg_2 = disparityregression(32,2)
        self.disp_reg_4 = disparityregression(self.max_disp//4,4)
        self.disp_reg_8 = disparityregression(self.max_disp//8,8)
        self.disp_reg_16 = disparityregression(self.max_disp//16,16)
    
    def cost_volume(self, refimg_fea, targetimg_fea, maxdisp, leftview=True):
        '''
        diff cost volume
        '''
        width = refimg_fea.shape[-1]
        mask_channel = refimg_fea.shape[1]
        # print(refimg_fea.size()[0])
        # print(mask_channel)
        # print(refimg_fea.size())
        cost = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1], maxdisp,  refimg_fea.size()[2],  refimg_fea.size()[3]).fill_(0.))
        full_mask = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0],refimg_fea.size()[1],refimg_fea.size()[2],refimg_fea.size()[3]).fill_(5.),requires_grad=False)

        for i in range(maxdisp):
            feata = refimg_fea[:,:,:,i:width]
            featb = targetimg_fea[:,:,:,:width-i]
            diff = torch.abs(feata - featb)
            #mask = full_mask[:,:,:,i:width] - diff
            #mask = torch.clamp(mask,0,5)
            #fullcost = feata * featb
            # concat
            if leftview:
                cost[:, :refimg_fea.size()[1], i, :,i:] = diff
            else:
                cost[:, :refimg_fea.size()[1], i, :,:width-i] = diff
        cost = cost.contiguous()

        return cost

    def forward(self,left,right):
        # print(image.size())
        left_features = self.feature_extraction(left)#origin size 64D test version
        right_features = self.feature_extraction(right)
        cost_volume_16 = self.cost_volume(left_features[3],right_features[3],self.max_disp//16)
        cost_volume_8 = self.cost_volume(left_features[2],right_features[2],self.max_disp//8)
        cost_volume_4 = self.cost_volume(left_features[1],right_features[1],self.max_disp//4)
        #print(cost_volume_16.size())
        cost_volume_16x2,cost_16 = self.decoder_16(cost_volume_16)
        disp_16 = self.disp_reg_16(F.softmax(cost_16,1))
        cost_volume_8 = torch.cat((cost_volume_16x2,cost_volume_8),1)
        
        cost_volume_8x2,cost_8 = self.decoder_8(cost_volume_8)
        disp_8 = self.disp_reg_8(F.softmax(cost_8,1))
        cost_volume_4 = torch.cat((cost_volume_4,cost_volume_8x2),1)

        cost_volume_4x2,cost_4 = self.decoder_4(cost_volume_4)
        disp_4 = self.disp_reg_4(F.softmax(cost_4,1))
        disp_4U = F.interpolate(disp_4,[left_features[0].size()[2],left_features[0].size()[3]],mode='bilinear')
        
        left_features_warp = warp(left_features[0],disp_4U)
        #print('total gpu memory %f M'%(Gpu_Memory(0)+Gpu_Memory(1)+Gpu_Memory(2)+Gpu_Memory(3)))
        cost_volume_2 = self.cost_volume(left_features_warp, right_features[0],32)
        _,cost_2 = self.decoder_2(cost_volume_2)#16*H*W
        disp_2 = self.disp_reg_2(F.softmax(cost_2,1))
        disp_2 = disp_2 + disp_4U
        #disp_1 = F.interpolate(disp_2,[left.size()[2],left.size()[3]],mode='bilinear')
        
        disp_16 = F.interpolate(disp_16,[left.size()[2],left.size()[3]],mode='bilinear')
        disp_8 = F.interpolate(disp_8,[left.size()[2],left.size()[3]],mode='bilinear')
        disp_4 = F.interpolate(disp_4,[left.size()[2],left.size()[3]],mode='bilinear')
        disp_2 = F.interpolate(disp_4,[left.size()[2],left.size()[3]],mode='bilinear')
        return disp_2,disp_4,disp_8,disp_16
