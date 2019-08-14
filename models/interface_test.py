# from __future__ import print_function
# import torch
# import torch.nn as nn
# import torch.utils.data
# from torch.autograd import Variable
# import torch.nn.functional as F
# import math
# from .submodule import *
# import sparseconvnet as scn
# #################
# #feature5 = 2048 H/64 too deep and useless
# #feature4 = 1024 H/32
# #feature3 = 512 H/16
# #feature2 = 256 H/8
# #feature1 = 64 H/4
# #feature0 = 64 H/2
# #################
# m = 16 # 16 or 32
# residual_blocks=False #True or False
# block_reps = 1 #Conv block repetition factor: 1 or 2
# class HRSCnet(nn.Module):
    
#     def __init__(self, max_disp):
#         super(HRSCnet, self).__init__()
#         self.max_disp = max_disp
#         #self.train_stage = train_stage
#         layer = [1,1,1,0]
#         self.feature_extraction = feature_extraction(layer)
        
#         # self.sparseModel = scn.Sequential().add(
#         #     scn.DenseToSparse(3).add(
#         #     scn.SubmanifoldConvolution(3, 3, m, 3, False)).add(
#         #         scn.UNet(data.dimension, block_reps, [m, 2*m, 3*m, 4*m], residual_blocks)).add(
#         #     scn.BatchNormReLU(m)).add(
#         #     scn.SubmanifoldConvolution(3, m, 1, 3, False)).add(
#         #     scn.SparseToDense(3,1)
#         #     )
#         self.sparseModel = scn.Sequential(
#             scn.DenseToSparse(3),
#             scn.SubmanifoldConvolution(3, 64, m, 3, False),
#             scn.BatchNormReLU(m),
#             scn.SubmanifoldConvolution(3, m, m, 3, False),
#             scn.BatchNormReLU(m),
#             scn.SubmanifoldConvolution(3, m, 1, 3, False),
#             scn.SparseToDense(3,1)
#         )
#         # self.sparseModel = scn.Sequential().add(
#         #     scn.DenseToSparse(3).add(
#         #     scn.SubmanifoldConvolution(3, 3, m, 3, False)).add(
#         #     scn.SubmanifoldConvolution(3, 3, m, 3, False)).add(
#         #     scn.BatchNormReLU(m)).add(
#         #     scn.SubmanifoldConvolution(3, m, 1, 3, False)).add(
#         #     scn.SparseToDense(3,1)
#         #     ))
#         self.disp_reg = disparityregression(self.max_disp,1)
    
#     def cost_volume(self, refimg_fea, targetimg_fea, maxdisp, leftview=True):
#         '''
#         diff cost volume
#         '''
#         width = refimg_fea.shape[-1]
#         mask_channel = refimg_fea.shape[1]
#         # print(refimg_fea.size()[0])
#         # print(mask_channel)
#         # print(refimg_fea.size())
#         Gpu_Memory()
#         cost = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1], maxdisp,  refimg_fea.size()[2],  refimg_fea.size()[3]).fill_(0.))
#         full_mask = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0],refimg_fea.size()[1],refimg_fea.size()[2],refimg_fea.size()[3]).fill_(5.),requires_grad=False)
#         Gpu_Memory()
#         for i in range(maxdisp):
#             feata = refimg_fea[:,:,:,i:width]
#             featb = targetimg_fea[:,:,:,:width-i]
#             diff = torch.abs(feata - featb)
#             mask = full_mask[:,:,:,i:width] - torch.abs(feata - featb)
#             mask = torch.clamp(mask,0,5)
#             fullcost = feata * featb
#             # concat
#             if leftview:
#                 cost[:, :refimg_fea.size()[1], i, :,i:] = fullcost * mask
#             else:
#                 cost[:, :refimg_fea.size()[1], i, :,:width-i] = fullcost * mask
#         cost = cost.contiguous()
#         Gpu_Memory()
#         return cost

#     def forward(self,left,right):
#         # print(image.size())
#         left_features = self.feature_extraction(left)#origin size 64D test version
#         right_features = self.feature_extraction(right)
#         # print(lr.size())
#         cost_volume = self.cost_volume(left_features, right_features, self.max_disp)#N*64*max_disp*H*W
#         cost_aggregation = self.sparseModel(cost_volume)
#         print(cost_aggregation.size())
#         cost_aggregation = F.interpolate(cost_aggregation, [self.disp_reg.disp.shape[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
#         print(cost_aggregation.size())
#         entrop = F.softmax(cost_aggregation,1)
#         print(entrop.size())
#         pred = self.disp_reg(entrop)

#         return pred

# class HRSCnet(nn.Module):
    
#     def __init__(self, max_disp):
#         super(HRSCnet, self).__init__()
#         self.max_disp = max_disp
#         #self.train_stage = train_stage
#         layer = [1,1,1,0]
#         self.feature_extraction = feature_extraction(layer)
        
#         self.sparseModel = scn.Sequential(
#             scn.DenseToSparse(3),
#             scn.SubmanifoldConvolution(3, 64, m, 3, False),
#             scn.BatchNormReLU(m),
#             scn.SubmanifoldConvolution(3, m, m, 3, False),
#             scn.BatchNormReLU(m),
#             scn.SubmanifoldConvolution(3, m, 1, 3, False),
#             scn.SparseToDense(3,1)
#         )
#         # self.sparseModel = scn.Sequential().add(
#         #     scn.DenseToSparse(3).add(
#         #     scn.SubmanifoldConvolution(3, 3, m, 3, False)).add(
#         #     scn.SubmanifoldConvolution(3, 3, m, 3, False)).add(
#         #     scn.BatchNormReLU(m)).add(
#         #     scn.SubmanifoldConvolution(3, m, 1, 3, False)).add(
#         #     scn.SparseToDense(3,1)
#         #     ))
#         self.disp_reg = disparityregression(self.max_disp,1)
    
#     def cost_volume(self, refimg_fea, targetimg_fea, maxdisp, leftview=True):
#         '''
#         diff cost volume
#         '''
#         width = refimg_fea.shape[-1]
#         mask_channel = refimg_fea.shape[1]
#         # print(refimg_fea.size()[0])
#         # print(mask_channel)
#         # print(refimg_fea.size())
#         Gpu_Memory()
#         cost = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1], maxdisp,  refimg_fea.size()[2],  refimg_fea.size()[3]).fill_(0.))
#         full_mask = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0],refimg_fea.size()[1],refimg_fea.size()[2],refimg_fea.size()[3]).fill_(5.),requires_grad=False)
#         Gpu_Memory()
#         for i in range(maxdisp):
#             feata = refimg_fea[:,:,:,i:width]
#             featb = targetimg_fea[:,:,:,:width-i]
#             diff = torch.abs(feata - featb)
#             mask = full_mask[:,:,:,i:width] - torch.abs(feata - featb)
#             mask = torch.clamp(mask,0,5)
#             fullcost = feata * featb
#             # concat
#             if leftview:
#                 cost[:, :refimg_fea.size()[1], i, :,i:] = fullcost * mask
#             else:
#                 cost[:, :refimg_fea.size()[1], i, :,:width-i] = fullcost * mask
#         cost = cost.contiguous()
#         Gpu_Memory()
#         return cost

#     def forward(self,left,right):
#         # print(image.size())
#         left_features = self.feature_extraction(left)#origin size 64D test version
#         right_features = self.feature_extraction(right)
#         # print(lr.size())
#         cost_volume = self.cost_volume(left_features, right_features, self.max_disp)#N*64*max_disp*H*W
#         cost_aggregation = self.sparseModel(cost_volume)
#         print(cost_aggregation.size())
#         cost_aggregation = F.interpolate(cost_aggregation, [self.disp_reg.disp.shape[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
#         print(cost_aggregation.size())
#         entrop = F.softmax(cost_aggregation,1)
#         print(entrop.size())
#         pred = self.disp_reg(entrop)

#         return pred