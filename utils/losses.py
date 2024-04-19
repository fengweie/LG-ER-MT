import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w, d = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

#def entropy_loss(p,C=2):
#    ## p N*C*W*H*D
#    y1 = -1*torch.sum(p*torch.log(p), dim=1)/torch.tensor(np.log(C)).cuda()
#    ent = torch.mean(y1)
#
#    return ent
def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w x d
        output: batch_size x 1 x h x w x d
    """
    assert v.dim() == 5
    n, c, h, w, d = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * d * np.log2(c))
def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
def Label2EdgeWeights(source_label):
    W = torch.zeros(source_label.shape[0], source_label.shape[0])
    Label = (source_label+1).float()
    Label = Label.unsqueeze(0)
    W = torch.transpose(1.0/Label, 1, 0)*Label
    W = torch.where(W!=1.0, torch.full_like(W, 0.), W)    
    return W
def discrimitive_loss(feature_embedding, input_logits):
    """compute discrimitive loss with the feature_embedding, output_logits

    Note:
    - Returns the discrimitive loss sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    # Random shuffle.
    np.random.seed()
    # input_logits = torch.max(input_logits,1)[1]
    x_value = np.random.randint(0,108)
    y_value = np.random.randint(0,108)
    z_value = np.random.randint(0,76)
    sub_feature_embedding = feature_embedding[:,:,x_value:x_value+5,y_value:y_value+5,z_value:z_value+5]
    sub_feature_embedding = sub_feature_embedding.contiguous().view(sub_feature_embedding.size(0),
                                                             sub_feature_embedding.size(1),-1)
    sub_input_logits = input_logits[:,x_value:x_value+5,y_value:y_value+5,z_value:z_value+5]
    sub_input_logits = sub_input_logits.contiguous().view(sub_input_logits.size(0),-1)
    discriminative_loss = 0
    for idx in range(sub_feature_embedding.size(0)):
        embedding = sub_feature_embedding[idx].transpose(1,0)
        logits = sub_input_logits[idx]
        total0 = embedding.unsqueeze(0).expand(int(embedding.size(0)), int(embedding.size(0)), int(embedding.size(1)))
        total1 = embedding.unsqueeze(1).expand(int(embedding.size(0)), int(embedding.size(0)), int(embedding.size(1)))
        emb_eucd = torch.norm((total0-total1),p=2,dim=2)
        neighbor_var = Label2EdgeWeights(logits)

        margin = 80
        F0 = torch.pow(emb_eucd, 2)
        F1 = torch.pow(torch.max(torch.full_like(emb_eucd, 0.0), margin-emb_eucd), 2)
        intra_loss = torch.mean(torch.mul(F0, neighbor_var))
        inter_loss = torch.mean(torch.mul(F1, 1.0-neighbor_var))
        sub_loss = (intra_loss + inter_loss) 
        
        discriminative_loss = discriminative_loss + sub_loss
    return discriminative_loss / sub_feature_embedding.size(0)    
    # np.random.seed()
    # # input_logits = torch.max(input_logits,1)[1]
    
    # feature_embedding = feature_embedding.view(feature_embedding.size(0),feature_embedding.size(1),-1)
    # input_logits = input_logits.view(input_logits.size(0),-1)
    
    # indices = np.arange(feature_embedding.size(2))
    # np.random.shuffle(indices)
   
    # feature_embedding = feature_embedding[:,:,indices]
    # input_logits = input_logits[:,indices]

    # emb_eucd = torch.norm((feature_embedding[:,:,:feature_embedding.size(2)//2]-feature_embedding[:,:,feature_embedding.size(2)//2:]),
    #                       p=2,dim=1) / 16
                         
    # neighbor_var = torch.eq(input_logits[:,:feature_embedding.size(2) // 2],
    #                     input_logits[:,feature_embedding.size(2) // 2:]).float()

    # margin = 1
    # pos = neighbor_var * torch.pow(emb_eucd, 2)
    # neg = (1. - neighbor_var) * torch.pow(torch.max(torch.full_like(emb_eucd, 0.0),
    #                                       margin - emb_eucd), 2)
    # discriminative_loss = torch.mean(pos + neg)
    # return discriminative_loss

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3],f_.shape[4]) + 1e-8

def similarity(feat):
    feat = feat.float()
    # tmp = L2(feat).detach()
    tmp = L2(feat)
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2]*f_T.shape[-3])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis
def CriterionPairWiseforWholeFeatAfterPool(preds_S, preds_T):
        feat_S = F.softmax(preds_S, dim=1)
        feat_T = F.softmax(preds_T, dim=1)
    
        loss = 0
        # maxpool = nn.AvgPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=0, ceil_mode=True) # change
        ix_all = np.random.choice(7, 2, replace=False)
        iy_all = np.random.choice(7, 2, replace=False)
        iz_all = np.random.choice(5, 1, replace=False)
#        ix_all = [1]
#        iy_all = [1]
#        iz_all = [1]
        for ix in ix_all:
            for iy in iy_all:
                for iz in iz_all:
                    sub_feat_S = feat_S[:,:,(ix*16):((ix+1)*16),(iy*16):((iy+1)*16),(iz*16):((iz+1)*16)]
                    sub_feat_T = feat_T[:,:,(ix*16):((ix+1)*16),(iy*16):((iy+1)*16),(iz*16):((iz+1)*16)]
                    
                    sub_loss = sim_dis_compute(sub_feat_S,sub_feat_T)
                    loss = loss + sub_loss
        return loss/(2*2)
        # np.random.seed()
        # # input_logits = torch.max(input_logits,1)[1]
        # x_value = np.random.randint(0,98)
        # y_value = np.random.randint(0,98)
        # z_value = np.random.randint(0,66)
        # feat_S = feat_S[:,:,x_value:x_value+15,y_value:y_value+15,z_value:z_value+15]
        # feat_T = feat_T[:,:,x_value:x_value+15,y_value:y_value+15,z_value:z_value+15]

    
        # total_w, total_h, total_D = feat_T.shape[2], feat_T.shape[3], feat_T.shape[4]
        # patch_w, patch_h, patch_D = int(total_w*self.scale), int(total_h*self.scale), int(total_D*self.scale)
        
        # # maxpool = nn.AvgPool3d(kernel_size=(patch_w, patch_h, patch_D), stride=(patch_w, patch_h, patch_D), padding=0, ceil_mode=True) # change
        # loss = self.criterion(feat_S,feat_T)
#        return loss
def selfinformation(preds_S, preds_T):
        feat_S = prob_2_entropy(F.softmax(preds_S, dim=1))
        feat_T = prob_2_entropy(F.softmax(preds_T, dim=1))
        
        loss = torch.mean((feat_S- feat_T)**2)/2

        return loss
