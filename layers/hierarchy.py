import torch
from torch import nn

import utils

import sources
from sources import Poisson
from .flow import Flow
#rom layers.resnet import ConditionalMapping
from layers.conditionalmapping import ConditionalMapping, SimpleConditionalMapping
from pdp_model import PoissonDirichlet
import numpy as np

from math import log
import matplotlib.pyplot as plt
#add poisson dirichlet process and mappings



class HierarchyBijector(Flow):
    def __init__(self, indexI, indexJ, latentI,latentJ,indexISeq,indexJSeq,indexKSeq,latSeqI,latSeqJ,latSeqK, layers,upper_layers,seq_depth, prior=None):
        super().__init__(prior)
        assert len(layers) == len(indexI)
        assert len(layers) == len(indexJ)
        self.seq_len = 4**(seq_depth-1) #expected planning length

        self.patch_size=4  #edge length of rg-blocks
        self.channels=1 #

        self.rate=10
        self.upper_layers = nn.ModuleList()



        self.layers = nn.ModuleList(layers)
        self.upper_layers = nn.ModuleList(upper_layers)
        self.indexI = indexI
        self.indexJ = indexJ
        self.latentI = latentI
        self.latentJ = latentJ
        self.indexISeq = indexISeq
        self.indexJSeq = indexJSeq
        self.indexKSeq = indexKSeq
        self.latentISeq = latSeqI
        self.latentJSeq = latSeqJ
        self.latentKSeq = latSeqK

        self.latents = []
        self.conditional_mappings = nn.ModuleList()
        self.PDP = nn.ModuleList()
        self.remove_num=0



    def organize_latents(self,x): #TODO will need something more robust for variable sequences
        x = utils.stackRGblock(x)

        K = self.patch_size
        C = self.channels
        x = x.view(-1, C, 4, K, K)  # shape: (batch_size*seq_len/4, 4, C, K, K)


        return x

    def organize_latents_inverse(self,z):#TODO will need something more robust for variable sequences

        #z = utils.stackRGblock(z)
        K=self.patch_size
        z = z.view(-1, self.channels, K,K)



        return z

    def forward_spatial(self,x):   #encode single image for online learning
        #dim(x) = (B,C,H,W)
        B, C, W, H = x.shape
        spatial_latents = []
        for i,(layer, indexI, indexJ,latI,latJ) in enumerate(zip(self.layers, self.indexI,  #forward through individual timestep layer
                                         self.indexJ,self.latentI,self.latentJ)):

            x, x_ = utils.dispatch(indexI, indexJ, x)

            x_ = utils.stackRGblock(x_)

            x_, log_prob = layer.forward(x_)

            x_ = utils.unstackRGblock(x_, B)

            x = utils.collect(indexI, indexJ, x, x_)
            if i % 2 == 1:
                _, latent = utils.dispatch_latents(latI,latJ,x)
                latent,_ = utils.exp_forward(latent)
                spatial_latents.append(latent.detach().cpu())
        return spatial_latents

    def forward_temporal(self,x): #x = list(frames)
        num_frames = len(x)

        if num_frames >=2:
            missing_frames = 4 - num_frames

            x = torch.stack([utils.stackRGblock(frame) for frame in x], dim=2)
            if missing_frames>0:
                padding = torch.ones((x.shape[0],x.shape[1],missing_frames,x.shape[3],x.shape[4])).to(x.device)
                x = torch.cat([x,padding],dim=2)

            B = x.shape[0]
            t_block = x.shape[2] / 2
            temporal_latents = []
            for i, (layer, indexI, indexJ, indexK, latI, latJ, latK) in enumerate(
                    zip(self.upper_layers, self.indexISeq, self.indexJSeq, self.indexKSeq, self.latentISeq, self.latentJSeq,
                        self.latentKSeq)):

                x, x_ = utils.dispatch_3d(indexK, indexI, indexJ, x)
                x_ = utils.stackRGblock_3d(x_, t_block, self.patch_size)
                x_, log_prob = layer.forward(x_)
                x_ = utils.unstackRGBlock_3d(x_, B)

                x = utils.collect_3d(indexK, indexI, indexJ, x, x_)
                if i % 2 == 1:
                    _, latent = utils.dispatch_3d(latK, latI, latJ, x)
                    latent, _ = utils.exp_forward(latent.clone())
                    temporal_latents.append(latent.detach().cpu())

        else:
            temporal_latents = [None for _ in range(len(self.upper_layers)//2)]

        if num_frames <4 and num_frames >=2:
            temporal_latents[-1] = None
            temporal_latents[0] = temporal_latents[0][:,:,0,:].unsqueeze(2) #get the only temporal observation for this layer
        elif num_frames>=4:
            temporal_latents[0] = temporal_latents[0][:,:,-1,:].unsqueeze(2) #get most recent temporal observation for this layer
        return temporal_latents


    def forward(self, x, train=False):
        # dim(x) = (B, C, H, W)
        device = x.device

        batch_size,seq_len,C,W,H = x.shape
        total_batch = batch_size * seq_len
        x = x.view(total_batch,C,W,H)


        ldj = x.new_zeros(total_batch)

        depth=0
        total_depth= len(self.layers)
        spatial_latents = []
        for i,(layer, indexI, indexJ,latI,latJ) in enumerate(zip(self.layers, self.indexI,  #forward through individual timestep layer
                                         self.indexJ,self.latentI,self.latentJ)):

            x, x_ = utils.dispatch(indexI, indexJ, x)

            x_ = utils.stackRGblock(x_)

            x_, log_prob = layer.forward(x_)

            ldj = ldj + log_prob.view(total_batch, -1).sum(dim=1)

            x_ = utils.unstackRGblock(x_, total_batch)

            x = utils.collect(indexI, indexJ, x, x_)
            if i % 2 == 1:
                _, latent = utils.dispatch_latents(latI,latJ,x)
                latent,_ = utils.exp_forward(latent.clone().detach())
                spatial_latents.append(latent)
            depth+=1

        x,ldj_exp = utils.exp_forward(x)
        x_,_ = utils.exp_forward(x_)
        ldj = ldj+ldj_exp
        lower_x = x
        upper_x = []
        upper_ldjs = []

        #x_ -> (total,C,K,K)
        x = self.organize_latents(x_.clone().detach().requires_grad_(True))

        #x -> (total/4,C,4,K,K)
        total_batch = x.shape[0]

        t_block = x.shape[2]/2
        upper_ldj = x.new_zeros(total_batch)



        test=0
        temporal_latents = []
        for i,(layer, indexI, indexJ,indexK,latI,latJ,latK) in enumerate(zip(self.upper_layers, self.indexISeq, self.indexJSeq,self.indexKSeq,self.latentISeq,self.latentJSeq,self.latentKSeq)):
            test+=1
            x, x_ = utils.dispatch_3d(indexK,indexI, indexJ, x)

            x_ = utils.stackRGblock_3d(x_,t_block,self.patch_size)

            x_, log_prob = layer.forward(x_)

            upper_ldj = upper_ldj + log_prob.view(total_batch, -1).sum(dim=1)

            x_ = utils.unstackRGBlock_3d(x_,total_batch)




            x = utils.collect_3d(indexK,indexI, indexJ, x, x_)
            if i % 2 == 1:
                _,latent = utils.dispatch_3d(latK,latI,latJ,x)
                latent,_ = utils.exp_forward(latent)
                temporal_latents.append(latent.clone().detach())

        x,ldj_exp = utils.exp_forward(x)


        upper_x.append(x)
        upper_ldjs.append(upper_ldj+ldj_exp)

        return lower_x, upper_x,ldj,upper_ldjs#, spatial_latents,temporal_latents

    def inverse(self, z_top, z_lower):
       
        B, C, T, H, W = z_top.shape

        inv_ldj = z_top.new_zeros(B)

        # ------------- A.  Invert the temporal (3-D) hierarchy -------------
        #
        # Iterate in *reverse* order through the 4 temporal RG phases:
        # (lists are length 4 after your fixes, so reversing them is enough)
        #

        for layer, idxK, idxI, idxJ in zip(
                reversed(self.upper_layers),  # RG blocks
                reversed(self.indexKSeq),
                reversed(self.indexISeq),
                reversed(self.indexJSeq)
        ):
           

            z_top, z_patch = utils.dispatch_3d(idxK, idxI, idxJ, z_top)  # (B,C,M,P)

         
            z_patch = utils.stackRGblock_3d(z_patch,
                                            T/2,  # 2
                                            patch_size=self.patch_size)  # K

           
            z_patch, ldj = layer.inverse(z_patch)  # same shape
            inv_ldj += ldj.view(B, -1).sum(1)

          
            z_patch = utils.unstackRGBlock_3d(z_patch,
                                              batch_size=B,
                                              )

      
            z_top = utils.collect_3d(idxK, idxI, idxJ, z_top, z_patch)

        z_top = self.organize_latents_inverse(z_top)

        idxI_bottom = self.indexI[-1]
        idxJ_bottom = self.indexJ[-1]

        B, C, L,_ = z_lower.shape


        z_lower = utils.collect(idxI_bottom, idxJ_bottom, z_lower, z_top)
     
        x = z_lower  # rename for clarity

        for i in reversed(range(len(self.layers))):
            idxI = self.indexI[i]
            idxJ = self.indexJ[i]

            x, x_patch = utils.dispatch(idxI, idxJ, x)  # (B,C,M,K²)
            x_patch = utils.stackRGblock(x_patch)  # (B·M,C,K,K)
            x_patch, ldj = self.layers[i].inverse(x_patch)

            #inv_ldj += ldj.view(B, -1).sum(1)
            x_patch = utils.unstackRGblock(x_patch, B)  # (B,C,M,K²)

            x = utils.collect(idxI, idxJ, x, x_patch)

        return x, inv_ldj

    def inverse_test(self,cluster):
        device = cluster.device
        batch_size = cluster.shape[0]
        z = torch.zeros(batch_size,1,32,32)
        inv_ldj = cluster.new_zeros(batch_size)

        for i in reversed(range(len(self.layers))):
            # if i %2 == 1 and i !=7:
            #     print(i,"LAYER")
            #     rate = torch.tensor(5.0).expand(int(z_.numel() * 3))
            #     latents = torch.poisson(rate)
            #     z = utils.collect_latents(self.latentI[i],self.latentJ[i],z.to(device),latents.to(device))

            z, z_ = utils.dispatch(self.indexI[i], self.indexJ[i], z)


            z_ = utils.stackRGblock(z_)
            if i == 0:
                z_ = self.PDP[-1].sample_from_clusters(torch.ones((16,), dtype=torch.int32) * cluster)

            z_, log_prob = self.layers[i].inverse(z_)

            inv_ldj = inv_ldj + log_prob.view(batch_size, -1).sum(dim=1)

            z_ = utils.unstackRGblock(z_, batch_size)
            z = utils.collect(self.indexI[i], self.indexJ[i], z, z_)



        return z, inv_ldj


