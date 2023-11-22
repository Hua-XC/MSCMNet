import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        #inputs(batch,2048)
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        ###################################################################

        dist=comp_dist(inputs,inputs)
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
    
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        #list->tensor (96)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        #dist——an tensor
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()

        return loss     
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
      
        dist_mat = comp_dist(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        #mask
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg
        #
        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        #(96)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss#, correct
        
def comp_dist(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    #torch.Size([96, 1024])
    m, n = emb1.shape[0], emb2.shape[0]
    #a2+b2
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    #dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    #(a-b)2
    dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
    
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx   

def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

class KLalignloss_DIMA(nn.Module):
    
    def __init__(self):
        super(KLalignloss_DIMA, self).__init__()

    def forward(self,p_logit, q_logit,align):
        if align=='channel':
            dima=1
        elif align=='height':
            dima=2
        elif align=='weight':
            dima=3
        elif align=='sample':
            dima=0
            
 
        kl = torch.sum(F.softmax(p_logit, dim=dima) * (F.log_softmax(p_logit, dim=dima)- F.log_softmax(q_logit, dim=dima)), 1)
        return torch.mean(kl)

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feat_dim, reduction='mean'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.reduction = reduction

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).to(device=x.device, dtype=torch.long)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        loss = distmat * mask.float()

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class CenterTripletLoss(nn.Module):
    def __init__(self, k_size, margin=0):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        #array
        centers = torch.stack(centers)
        # centers:torch.Size([96, 2048]) input：torch.Size([96, 2048])       
        dist_pc = (inputs - centers)**2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, centers, centers.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_an, dist_ap = [], []
        for i in range(0, n, self.k_size):
            dist_an.append((self.margin - dist[i][mask[i] == 0]).clamp(min=0.0).mean() )
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        loss = dist_pc.mean() + dist_an.mean()
        return loss/2#, dist_pc.mean(), dist_an.mean()

class QuarCenterTripletLoss(nn.Module):
    def __init__(self, k_size, margin=0,t=0):
        super(QuarCenterTripletLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        
        targetsIND=targets[0:n//2]
        inputsRGBD=inputs[0:n//2]
        targetsIND=targets[n//2:n]
        inputsIRD=inputs[n//2:n]

        # Come to centers
        #rgb1和ir1---rgb2和ir2
        centers = []
        centers_dual=[]
        centers_dualIR1 = []
        centers_dualIR2 = []
        centers_dualRGB1 = []
        centers_dualRGB2 = []
        
        targetsIN=targets[0:n//4]

        inputsRGB1=inputs[0:n//4]
        inputsRGB2=inputs[n//4:n//2]
    
        inputsIR1=inputs[n//2:3*n//4]
        inputsIR2=inputs[3*n//4:n]
        
        for i in range(n//4):
            #quadruple
            centers_dualIR1.append(inputsIR1[targetsIN == targetsIN[i]].mean(0))
            centers_dualIR2.append(inputsIR2[targetsIN == targetsIN[i]].mean(0))
            centers_dualRGB1.append(inputsRGB1[targetsIN == targetsIN[i]].mean(0))
            centers_dualRGB2.append(inputsRGB2[targetsIN == targetsIN[i]].mean(0))
        for i in range(n//2):
            centers_dual.append(inputsIRD[targetsIND == targetsIND[i]].mean(0))
        for i in range(n//2):
            centers_dual.append(inputsRGBD[targetsIND == targetsIND[i]].mean(0))
        
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        centers = torch.stack(centers)
        centers_dual = torch.stack(centers_dual)     

        centers_dualIR1 = torch.stack(centers_dualIR1)#IR1
        centers_dualIR2 = torch.stack(centers_dualIR2)#IR2   
        centers_dualRGB1 = torch.stack(centers_dualRGB1)# RGB1
        centers_dualRGB2 = torch.stack(centers_dualRGB2)#RGB2
        
        dist_pc_dual = (inputs - centers_dual)**2#+(inputs - centers)**2
        #center_quar=torch.cat([centers_dualIR2,centers_dualIR1,centers_dualRGB2,centers_dualRGB1])
        center_quar=torch.cat([centers_dualRGB1,centers_dualRGB2,centers_dualIR1,centers_dualIR2])
        #inter——class
        #dist_quar=(centers_dualRGB1-centers_dualRGB2)**2+(centers_dualIR1-centers_dualIR2)**2
        #dist_quar=(centers_dualRGB1-centers_dualIR1)**2+(centers_dualRGB2-centers_dualIR2)**2
        dist_quar=(inputs-center_quar)**2
        #set the alpha
        a=0
        dist_quaro = a*dist_quar.sum(1).sqrt()
        dist_pc = (1-a)*dist_pc_dual.sum(1).sqrt()
        
  

        dist = comp_dist(inputs,centers)
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_an, dist_ap = [], []
        for i in range(0, n):
            dist_an.append((self.margin - dist[i][mask[i] == 0]).clamp(min=0.0).mean() )
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        loss = dist_pc.mean() + dist_an.mean()+dist_quaro.mean()
        return loss/2 #,dist_pc.mean(), dist_an.mean()

 
        