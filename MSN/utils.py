import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
import math

def criterion(ptg, pon, lamb=0.1):
    
    CELoss = torch.mean(torch.sum(torch.log(pon**(-ptg)), dim=1))
    
    avg_probs = torch.mean(pon, dim=0)
    rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))

    return CELoss + lamb*rloss

def get_byol_transforms(size, mean, std):
    transformT = tr.Compose([
    tr.RandomResizedCrop(size=size, scale=(0.08,1), ratio=(3 / 4, 4 / 3)),
    tr.RandomApply(nn.ModuleList([tr.RandomRotation((-90, 90))]), p=0.5),
    tr.RandomApply(nn.ModuleList([tr.ColorJitter()]), p=0.8),
    tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
    #tr.RandomGrayscale(p=0.2),
    tr.Normalize(mean, std)])

    transformT1 = tr.Compose([
        tr.RandomResizedCrop(size=size, scale=(0.08,1), ratio=(3 / 4, 4 / 3)),
        tr.RandomApply(nn.ModuleList([tr.RandomRotation((-90, 90))]), p=0.5),
        tr.RandomApply(nn.ModuleList([tr.ColorJitter()]), p=0.8),
        #tr.RandomGrayscale(p=0.2),
        tr.RandomApply(nn.ModuleList([tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0))]), p=0.1),
        tr.Normalize(mean, std)])

    transformEvalT = tr.Compose([
        tr.CenterCrop(size=size),
        tr.Normalize(mean, std)
    ])

    return transformT, transformT1, transformEvalT