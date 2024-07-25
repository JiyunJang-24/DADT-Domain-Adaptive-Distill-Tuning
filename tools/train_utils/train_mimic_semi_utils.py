import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.nn.utils import clip_grad_norm_
from pcdet.models import load_data_to_gpu
import pdb
import numpy as np

def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(p, m) + F.kl_div(q, m))


def cal_linear_probing(batch_teacher, batch):
    teacher_features = batch_teacher['bev_feature_projection'].detach()
    student_features = batch['bev_feature_projection']
    # teacher_features = F.normalize(teacher_features, p=2, dim=1)
    # student_features = F.normalize(student_features, p=2, dim=1)
    # cosine_sim = F.cosine_similarity(teacher_features, student_features)
    # loss = 1.0 - cosine_sim.mean()
    squared_errors = (teacher_features - student_features)**2
    # # 평균 제곱 오차 계산
    mse = torch.mean(squared_errors)
    return mse
    


def cal_mimic_loss(batch_teacher, batch, mimic_mode):
    teacher_features = batch_teacher['spatial_features_2d'].detach()
    student_features = batch['spatial_features_2d']

    if mimic_mode == 'gt':
        rois = batch['gt_boxes']
    # else:
    #     rois = batch_teacher['rois_mimic'].detach()
    #pdb.set_trace()
    dataset_cfg = batch['dataset_cfg']
    min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
    min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
    voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
    voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
    down_sample_ratio = 8


    if mimic_mode in ['roi', 'gt']:
        batch_size, height, width = teacher_features.size(0), teacher_features.size(2), teacher_features.size(3)
        roi_size = rois.size(1)

        x1 = (rois[:, :, 0] - rois[:, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
        x2 = (rois[:, :, 0] + rois[:, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
        y1 = (rois[:, :, 1] - rois[:, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
        y2 = (rois[:, :, 1] + rois[:, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
        #print(height, width,x1.min(),x2.max(),y1.min(),y2.max())
        mask = torch.zeros(batch_size, roi_size, height, width).bool().cuda()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        grid_y = grid_y[None, None].repeat(batch_size, roi_size, 1, 1).cuda()
        grid_x = grid_x[None, None].repeat(batch_size, roi_size, 1, 1).cuda()

        mask_y = (grid_y >= y1[:, :, None, None]) * (grid_y <= y2[:, :, None, None])
        mask_x = (grid_x >= x1[:, :, None, None]) * (grid_x <= x2[:, :, None, None])
        mask = (mask_y * mask_x).float()
        if mimic_mode == 'gt':
            mask[rois[:,:,-1] == 0] = 0
        weight = mask.sum(-1).sum(-1) #bz * roi
        weight[weight == 0] = 1
        mask = mask / weight[:, :, None, None]

        mimic_loss = torch.norm(teacher_features - student_features, p=2, dim=1)
        #mimic_loss = 1 - F.cosine_similarity(teacher_features, student_features, dim=1)
        #mimic_loss = jensen_shannon_divergence(F.normalize(teacher_features, p=2, dim=1),\
        #                                     F.normalize(student_features, p=2, dim=1))
        mask = mask.sum(1)
        mimic_loss = (mimic_loss * mask).sum() / batch_size / roi_size
        if mimic_mode == 'gt':
            mimic_loss = (mimic_loss * mask).sum() / (rois[:,:,-1] > 0).sum()
    elif mimic_mode == 'all':
        mimic_loss = torch.mean(torch.norm(teacher_features - student_features, p=2, dim=1))
        #mimic_loss = torch.mean(1 - F.cosine_similarity(teacher_features, student_features, dim=1))
        #mimic_loss = jensen_shannon_divergence(F.normalize(teacher_features, p=2, dim=1),\
        #                                     F.normalize(student_features, p=2, dim=1))
    else:
        raise NotImplementedError

    return mimic_loss


def cal_inter_class_loss(batch, model_cfg, temperature=0.07, contrast_mode='all', base_temperature=0.07):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...]. 292 * 1 * 512
        labels: ground truth of shape [bsz]. 292
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    
    # Extract features from the student model
    student_features = batch['spatial_features_2d']

    # Extract ground truth class information from the batch
    
    dataset_cfg = batch['dataset_cfg']
    min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
    min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
    max_x = dataset_cfg.POINT_CLOUD_RANGE[3]
    max_y = dataset_cfg.POINT_CLOUD_RANGE[4]
    voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
    voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
    down_sample_ratio = model_cfg.ROI_GRID_POOL.DOWNSAMPLE_RATIO

    # Extract region of interests (rois) from the batch
    rois = batch['gt_boxes']
    original_rois_shape2 = rois.shape
    original_rois_shape = (rois.shape[0] * rois.shape[1], rois.shape[2]) 
    selected_rois = rois[(rois[:, :, 0] > min_x) & (rois[:, :, 1] > min_y) & (rois[:, :, 0] < max_x) & (rois[:, :, 1] < max_y)]
    filled_rois = torch.zeros(original_rois_shape, dtype=rois.dtype, device=rois.device)
    if original_rois_shape != selected_rois.shape:
        filled_rois[:selected_rois.shape[0], :] = selected_rois
    # original_rois_shape과 동일한 크기의 텐서를 생성하고 0으로 초기화
    rois = filled_rois.view(original_rois_shape2)
    try:
        gt_class = rois[:, :, 7]
    except:
        gt_class = rois[:, 7]
    # Extract configuration information from the dataset_cfg in the batch
    # Get dimensions of the features
    batch_size, height, width = student_features.size(0), student_features.size(2), student_features.size(3)

    # Get the number of rois
    roi_size = rois.size(1)

    class_features_dict = {}

    # Iterate over instances
    for i in range(batch_size):
        # Iterate over rois for the current instance
        for j in range(roi_size):
            # Get the current roi's class label
            class_label = int(gt_class[i, j].item())

            # Skip instances with class label 0
            if class_label == 0:
                continue

            # Calculate normalized coordinates of the current roi for the current instance
            x1 = (rois[i, j, 0] - rois[i, j, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            x2 = (rois[i, j, 0] + rois[i, j, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            y1 = (rois[i, j, 1] - rois[i, j, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            y2 = (rois[i, j, 1] + rois[i, j, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)

            # Convert normalized coordinates to pixel coordinates
            x1_pixel = (x1).clamp(0, width - 1).long()
            x2_pixel = (x2+1).clamp(0, width - 1).long()
            y1_pixel = (y1).clamp(0, height - 1).long()
            y2_pixel = (y2+1).clamp(0, height - 1).long()

            # Extract region feature map from student_features
            region_feature = student_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
            region_feature = region_feature.mean(dim=(1, 2))
            # Initialize list for the current roi if not present
            if class_label not in class_features_dict:
                class_features_dict[class_label] = []

            # Add the region feature map to the list for the current roi and its associated class label
            class_features_dict[class_label].append(region_feature)
            
    features = F.normalize(torch.cat([torch.stack(class_features) for class_features in class_features_dict.values()]).unsqueeze(1).float(), dim=2)
    
    labels = torch.cat([torch.full((len(class_features),), class_label) for class_label, class_features in class_features_dict.items()]).unsqueeze(1)
    mask = None  # Since you're not using the mask in the provided example
    
    count_1 = (labels == 1).sum().item()
    count_2 = (labels == 2).sum().item()
    count_3 = (labels == 3).sum().item()
    if count_1 == 1 or count_2 == 1 or count_3 == 1:
        pdb.set_trace()
        
    
    
    device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)
    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    # modified to handle edge cases when there is no positive pair
    # for an anchor point. 
    # Edge case e.g.:- 
    # features of shape: [4,1,...]
    # labels:            [0,1,1,2]
    # loss before mean:  [nan, ..., ..., nan] 
    mask_pos_pairs = mask.sum(1)
    #mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    
    loss = loss.view(anchor_count, batch_size).mean()

    return loss

        
def cal_bev_attention_loss2_no_roi(batch_teacher, batch, index = None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...]. 292 * 1 * 512
        labels: ground truth of shape [bsz]. 292
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    loss = torch.tensor([0], dtype=torch.float32).cuda()
    
    teacher_features = batch_teacher['spatial_features_2d'].detach()
    # Extract features from the student model
    student_features = batch['spatial_features_2d']
    # student_features = batch['spatial_features_2d_group'][:, index]
    # Extract ground truth class information from the batch
    
    dataset_cfg = batch['dataset_cfg']
    min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
    min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
    max_x = dataset_cfg.POINT_CLOUD_RANGE[3]
    max_y = dataset_cfg.POINT_CLOUD_RANGE[4]
    voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
    voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
    
    # Extract region of interests (rois) from the batch
    rois = batch['gt_boxes']
    original_rois_shape2 = rois.shape
    original_rois_shape = (rois.shape[0] * rois.shape[1], rois.shape[2]) 
    selected_rois = rois[(rois[:, :, 0] > min_x) & (rois[:, :, 1] > min_y) & (rois[:, :, 0] < max_x) & (rois[:, :, 1] < max_y)]
    filled_rois = torch.zeros(original_rois_shape, dtype=rois.dtype, device=rois.device)
    if original_rois_shape != selected_rois.shape:
        filled_rois[:selected_rois.shape[0], :] = selected_rois
        rois = filled_rois.view(original_rois_shape2)
    # original_rois_shape과 동일한 크기의 텐서를 생성하고 0으로 초기화

    try:
        gt_class = rois[:, :, 7]
    except:
        gt_class = rois[:, 7]
        
    # Extract configuration information from the dataset_cfg in the batch
    # Get dimensions of the features
    batch_size, channel, height, width = teacher_features.size(0), teacher_features.size(1), teacher_features.size(2), teacher_features.size(3)
    x_down_sample_ratio = ((max_x - min_x) / voxel_size_x) / width
    y_down_sample_ratio = ((max_y - min_y) / voxel_size_y) / height
    # Get the number of rois    
    roi_size = rois.size(1)

    
    batch_class_features_dict = {} # [batch][class][i] = features, x1, x2, y1, y2
    batch_teacher_class_features_dict = {} # [batch][class][i] = features, x1, x2, y1, y2
    # Iterate over instances
    for i in range(batch_size):
        # Iterate over rois for the current instance
        student_features_batch = student_features[i]
        teacher_features_batch = teacher_features[i]
        for j in range(roi_size):
            # Get the current roi's class label
            class_label = int(gt_class[i, j].item())

            # Skip instances with class label 0
            if class_label != 1 and class_label != 2 and class_label != 3:
                continue

            # Calculate normalized coordinates of the current roi for the current instance
            x1 = (rois[i, j, 0] - rois[i, j, 3] / 2 - min_x) / (voxel_size_x * x_down_sample_ratio)
            x2 = (rois[i, j, 0] + rois[i, j, 3] / 2 - min_x) / (voxel_size_x * x_down_sample_ratio)
            y1 = (rois[i, j, 1] - rois[i, j, 4] / 2 - min_y) / (voxel_size_y * y_down_sample_ratio)
            y2 = (rois[i, j, 1] + rois[i, j, 4] / 2 - min_y) / (voxel_size_y * y_down_sample_ratio)

            # Convert normalized coordinates to pixel coordinates
            x1_pixel = (x1).clamp(0, width - 1).long()
            x2_pixel = (x2+1).clamp(0, width - 1).long()
            y1_pixel = (y1).clamp(0, height - 1).long()
            y2_pixel = (y2+1).clamp(0, height - 1).long()

            # Extract region feature map from student_features
            
            if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
                region_feature = student_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
                region_feature_teacher = teacher_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
            elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
                region_feature = student_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
                region_feature_teacher = teacher_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
            elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
                region_feature = student_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
                region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
            else:
                region_feature = student_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
                region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
            
            region_feature = region_feature.mean(dim=(1, 2))
            region_feature_teacher = region_feature_teacher.mean(dim=(1, 2))
            # Initialize list for the current roi if not present
            if  torch.isnan(region_feature).any().item():
                pdb.set_trace()
            
            if i not in batch_class_features_dict:
                batch_class_features_dict[i] = {}
                batch_teacher_class_features_dict[i] = {}
            if class_label not in batch_class_features_dict[i]:
                batch_class_features_dict[i][class_label] = []
                batch_teacher_class_features_dict[i][class_label] = []

            # Add the region feature map to the list for the current roi and its associated class label
            batch_class_features_dict[i][class_label].append([region_feature, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
            batch_teacher_class_features_dict[i][class_label].append([region_feature_teacher, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
            
        
        #batch = i
        if i not in batch_class_features_dict.keys() or i not in batch_teacher_class_features_dict.keys():
            continue
        
        for label in batch_class_features_dict[i].keys():
            bev_feature_class_gt = torch.zeros(height, width).cuda()
            class_key = torch.tensor([]).cuda()
            class_teacher_key = torch.tensor([]).cuda()
            
            for k in range(len(batch_class_features_dict[i][label])):
                class_key = torch.cat((class_key, batch_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
                class_teacher_key = torch.cat((class_teacher_key, batch_teacher_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
                
                # x1_pixel, x2_pixel, y1_pixel, y2_pixel = batch_class_features_dict[i][label][k][1:]
                
                # if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
                #     bev_feature_class_gt[y1_pixel, x1_pixel] = 1
                # elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
                #     bev_feature_class_gt[y1_pixel, x1_pixel:x2_pixel] = 1
                # elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
                #     bev_feature_class_gt[y1_pixel:y2_pixel, x1_pixel] = 1
                # else:
                #     bev_feature_class_gt[y1_pixel:y2_pixel, x1_pixel:x2_pixel] = 1
            
            kq_cross = F.normalize(torch.matmul(class_teacher_key, student_features_batch.reshape(channel, -1)), dim=0).cuda()
            #kq_cross = torch.matmul(class_teacher_key, student_features_batch.reshape(channel, -1)).cuda()
            kq_cross = kq_cross.view(class_teacher_key.shape[0], height, width)
            kq_self = F.normalize(torch.matmul(class_key, student_features_batch.reshape(channel, -1)), dim=0).cuda()
            #kq_self = torch.matmul(class_key, student_features_batch.reshape(channel, -1)).cuda()
            kq_self = kq_self.view(class_key.shape[0], height, width)
            mse_loss = 0
            weight = 1
            squared_errors = ((kq_cross - kq_self)*weight)**2
            mse = torch.mean(squared_errors)
            mse /= class_key.shape[0]
            loss += mse
            
            # for l in range(class_key.shape[0]):
            #     squared_errors = (bev_feature_class_gt - kq_cross[l])**2
            # # # 평균 제곱 오차 계산
            #     mse = torch.mean(squared_errors)
            #     mse_loss += mse
            # loss += (mse_loss / class_key.shape[0])
            
        #loss /= len(batch_class_features_dict[i].keys())
    
    loss /= batch_size
    #pdb.set_trace()
    return loss


        
def cal_bev_attention_loss2_no_roi_self(batch_teacher, batch, index = None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...]. 292 * 1 * 512
        labels: ground truth of shape [bsz]. 292
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    loss = torch.tensor([0], dtype=torch.float32).cuda()
    
    teacher_features = batch_teacher['spatial_features_2d'].detach()
    # Extract features from the student model
    student_features = batch['spatial_features_2d']
    # student_features = batch['spatial_features_2d_group'][:, index]
    # Extract ground truth class information from the batch
    
    dataset_cfg = batch['dataset_cfg']
    min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
    min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
    max_x = dataset_cfg.POINT_CLOUD_RANGE[3]
    max_y = dataset_cfg.POINT_CLOUD_RANGE[4]
    voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
    voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
    
    # Extract region of interests (rois) from the batch
    rois = batch['gt_boxes']
    original_rois_shape2 = rois.shape
    original_rois_shape = (rois.shape[0] * rois.shape[1], rois.shape[2]) 
    selected_rois = rois[(rois[:, :, 0] > min_x) & (rois[:, :, 1] > min_y) & (rois[:, :, 0] < max_x) & (rois[:, :, 1] < max_y)]
    filled_rois = torch.zeros(original_rois_shape, dtype=rois.dtype, device=rois.device)
    if original_rois_shape != selected_rois.shape:
        filled_rois[:selected_rois.shape[0], :] = selected_rois
        rois = filled_rois.view(original_rois_shape2)
    # original_rois_shape과 동일한 크기의 텐서를 생성하고 0으로 초기화

    try:
        gt_class = rois[:, :, 7]
    except:
        gt_class = rois[:, 7]
        
    # Extract configuration information from the dataset_cfg in the batch
    # Get dimensions of the features
    batch_size, channel, height, width = teacher_features.size(0), teacher_features.size(1), teacher_features.size(2), teacher_features.size(3)
    x_down_sample_ratio = ((max_x - min_x) / voxel_size_x) / width
    y_down_sample_ratio = ((max_y - min_y) / voxel_size_y) / height
    # Get the number of rois    
    roi_size = rois.size(1)

    
    batch_class_features_dict = {} # [batch][class][i] = features, x1, x2, y1, y2
    batch_teacher_class_features_dict = {} # [batch][class][i] = features, x1, x2, y1, y2
    # Iterate over instances
    for i in range(batch_size):
        # Iterate over rois for the current instance
        student_features_batch = student_features[i]
        teacher_features_batch = teacher_features[i]
        for j in range(roi_size):
            # Get the current roi's class label
            class_label = int(gt_class[i, j].item())

            # Skip instances with class label 0
            if class_label != 1 and class_label != 2 and class_label != 3:
                continue

            # Calculate normalized coordinates of the current roi for the current instance
            x1 = (rois[i, j, 0] - rois[i, j, 3] / 2 - min_x) / (voxel_size_x * x_down_sample_ratio)
            x2 = (rois[i, j, 0] + rois[i, j, 3] / 2 - min_x) / (voxel_size_x * x_down_sample_ratio)
            y1 = (rois[i, j, 1] - rois[i, j, 4] / 2 - min_y) / (voxel_size_y * y_down_sample_ratio)
            y2 = (rois[i, j, 1] + rois[i, j, 4] / 2 - min_y) / (voxel_size_y * y_down_sample_ratio)

            # Convert normalized coordinates to pixel coordinates
            x1_pixel = (x1).clamp(0, width - 1).long()
            x2_pixel = (x2+1).clamp(0, width - 1).long()
            y1_pixel = (y1).clamp(0, height - 1).long()
            y2_pixel = (y2+1).clamp(0, height - 1).long()

            # Extract region feature map from student_features
            
            if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
                region_feature = student_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
                region_feature_teacher = teacher_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
            elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
                region_feature = student_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
                region_feature_teacher = teacher_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
            elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
                region_feature = student_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
                region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
            else:
                region_feature = student_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
                region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
            
            region_feature = region_feature.mean(dim=(1, 2))
            region_feature_teacher = region_feature_teacher.mean(dim=(1, 2))
            # Initialize list for the current roi if not present
            if  torch.isnan(region_feature).any().item():
                pdb.set_trace()
            
            if i not in batch_class_features_dict:
                batch_class_features_dict[i] = {}
                batch_teacher_class_features_dict[i] = {}
            if class_label not in batch_class_features_dict[i]:
                batch_class_features_dict[i][class_label] = []
                batch_teacher_class_features_dict[i][class_label] = []

            # Add the region feature map to the list for the current roi and its associated class label
            batch_class_features_dict[i][class_label].append([region_feature, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
            batch_teacher_class_features_dict[i][class_label].append([region_feature_teacher, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
            
        
        #batch = i
        if i not in batch_class_features_dict.keys() or i not in batch_teacher_class_features_dict.keys():
            continue
        
        for label in batch_class_features_dict[i].keys():
            bev_feature_class_gt = torch.zeros(height, width).cuda()
            class_key = torch.tensor([]).cuda()
            class_teacher_key = torch.tensor([]).cuda()
            
            for k in range(len(batch_class_features_dict[i][label])):
                class_key = torch.cat((class_key, batch_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
                class_teacher_key = torch.cat((class_teacher_key, batch_teacher_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
                
                # x1_pixel, x2_pixel, y1_pixel, y2_pixel = batch_class_features_dict[i][label][k][1:]
                
                # if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
                #     bev_feature_class_gt[y1_pixel, x1_pixel] = 1
                # elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
                #     bev_feature_class_gt[y1_pixel, x1_pixel:x2_pixel] = 1
                # elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
                #     bev_feature_class_gt[y1_pixel:y2_pixel, x1_pixel] = 1
                # else:
                #     bev_feature_class_gt[y1_pixel:y2_pixel, x1_pixel:x2_pixel] = 1
            
            kq_cross = F.normalize(torch.matmul(class_teacher_key, teacher_features_batch.reshape(channel, -1)), dim=0).cuda()
            #kq_cross = torch.matmul(class_teacher_key, student_features_batch.reshape(channel, -1)).cuda()
            kq_cross = kq_cross.view(class_teacher_key.shape[0], height, width)
            kq_self = F.normalize(torch.matmul(class_key, student_features_batch.reshape(channel, -1)), dim=0).cuda()
            #kq_self = torch.matmul(class_key, student_features_batch.reshape(channel, -1)).cuda()
            kq_self = kq_self.view(class_key.shape[0], height, width)
            mse_loss = 0
            weight = 1
            squared_errors = ((kq_cross - kq_self)*weight)**2
            mse = torch.mean(squared_errors)
            mse /= class_key.shape[0]
            loss += mse
            
            # for l in range(class_key.shape[0]):
            #     squared_errors = (bev_feature_class_gt - kq_cross[l])**2
            # # # 평균 제곱 오차 계산
            #     mse = torch.mean(squared_errors)
            #     mse_loss += mse
            # loss += (mse_loss / class_key.shape[0])
            
        #loss /= len(batch_class_features_dict[i].keys())
    
    loss /= batch_size
    #pdb.set_trace()
    return loss


def cal_bev_attention_loss2_no_roi_add(batch_teacher, batch, index = None, mode = None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...]. 292 * 1 * 512
        labels: ground truth of shape [bsz]. 292
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    loss = torch.tensor([0], dtype=torch.float32).cuda()
    loss2 = torch.tensor([0], dtype=torch.float32).cuda()
    teacher_features = batch_teacher['spatial_features_2d'].detach()
    # Extract features from the student model
    student_features = batch['spatial_features_2d']
    # student_features = batch['spatial_features_2d_group'][:, index]
    # Extract ground truth class information from the batch
    
    dataset_cfg = batch['dataset_cfg']
    min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
    min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
    max_x = dataset_cfg.POINT_CLOUD_RANGE[3]
    max_y = dataset_cfg.POINT_CLOUD_RANGE[4]
    voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
    voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
    
    # Extract region of interests (rois) from the batch
    rois = batch['gt_boxes']
    original_rois_shape2 = rois.shape
    original_rois_shape = (rois.shape[0] * rois.shape[1], rois.shape[2]) 
    selected_rois = rois[(rois[:, :, 0] > min_x) & (rois[:, :, 1] > min_y) & (rois[:, :, 0] < max_x) & (rois[:, :, 1] < max_y)]
    filled_rois = torch.zeros(original_rois_shape, dtype=rois.dtype, device=rois.device)
    if original_rois_shape != selected_rois.shape:
        filled_rois[:selected_rois.shape[0], :] = selected_rois
        rois = filled_rois.view(original_rois_shape2)
    # original_rois_shape과 동일한 크기의 텐서를 생성하고 0으로 초기화

    try:
        gt_class = rois[:, :, 7]
    except:
        gt_class = rois[:, 7]
        
    # Extract configuration information from the dataset_cfg in the batch
    # Get dimensions of the features
    batch_size, channel, height, width = teacher_features.size(0), teacher_features.size(1), teacher_features.size(2), teacher_features.size(3)
    x_down_sample_ratio = ((max_x - min_x) / voxel_size_x) / width
    y_down_sample_ratio = ((max_y - min_y) / voxel_size_y) / height
    # Get the number of rois    
    roi_size = rois.size(1)

    
    batch_class_features_dict = {} # [batch][class][i] = features, x1, x2, y1, y2
    batch_teacher_class_features_dict = {} # [batch][class][i] = features, x1, x2, y1, y2
    # Iterate over instances
    for i in range(batch_size):
        # Iterate over rois for the current instance
        student_features_batch = student_features[i]
        teacher_features_batch = teacher_features[i]
        for j in range(roi_size):
            # Get the current roi's class label
            class_label = int(gt_class[i, j].item())

            # Skip instances with class label 0
            if class_label != 1 and class_label != 2 and class_label != 3:
                continue

            # Calculate normalized coordinates of the current roi for the current instance
            x1 = (rois[i, j, 0] - rois[i, j, 3] / 2 - min_x) / (voxel_size_x * x_down_sample_ratio)
            x2 = (rois[i, j, 0] + rois[i, j, 3] / 2 - min_x) / (voxel_size_x * x_down_sample_ratio)
            y1 = (rois[i, j, 1] - rois[i, j, 4] / 2 - min_y) / (voxel_size_y * y_down_sample_ratio)
            y2 = (rois[i, j, 1] + rois[i, j, 4] / 2 - min_y) / (voxel_size_y * y_down_sample_ratio)

            # Convert normalized coordinates to pixel coordinates
            x1_pixel = (x1).clamp(0, width - 1).long()
            x2_pixel = (x2+1).clamp(0, width - 1).long()
            y1_pixel = (y1).clamp(0, height - 1).long()
            y2_pixel = (y2+1).clamp(0, height - 1).long()

            # Extract region feature map from student_features
            
            if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
                region_feature = student_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
                region_feature_teacher = teacher_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
            elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
                region_feature = student_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
                region_feature_teacher = teacher_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
            elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
                region_feature = student_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
                region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
            else:
                region_feature = student_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
                region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
            
            region_feature = region_feature.mean(dim=(1, 2))
            region_feature_teacher = region_feature_teacher.mean(dim=(1, 2))
            # Initialize list for the current roi if not present
            if  torch.isnan(region_feature).any().item():
                pdb.set_trace()
            
            if i not in batch_class_features_dict:
                batch_class_features_dict[i] = {}
                batch_teacher_class_features_dict[i] = {}
            if class_label not in batch_class_features_dict[i]:
                batch_class_features_dict[i][class_label] = []
                batch_teacher_class_features_dict[i][class_label] = []

            # Add the region feature map to the list for the current roi and its associated class label
            batch_class_features_dict[i][class_label].append([region_feature, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
            batch_teacher_class_features_dict[i][class_label].append([region_feature_teacher, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
            
        
        #batch = i
        if i not in batch_class_features_dict.keys() or i not in batch_teacher_class_features_dict.keys():
            continue
        
        for label in batch_class_features_dict[i].keys():
            bev_feature_class_gt = torch.zeros(height, width).cuda()
            class_key = torch.tensor([]).cuda()
            class_teacher_key = torch.tensor([]).cuda()
            
            for k in range(len(batch_class_features_dict[i][label])):
                class_key = torch.cat((class_key, batch_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
                class_teacher_key = torch.cat((class_teacher_key, batch_teacher_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
                
                # x1_pixel, x2_pixel, y1_pixel, y2_pixel = batch_class_features_dict[i][label][k][1:]
                
                # if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
                #     bev_feature_class_gt[y1_pixel, x1_pixel] = 1
                # elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
                #     bev_feature_class_gt[y1_pixel, x1_pixel:x2_pixel] = 1
                # elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
                #     bev_feature_class_gt[y1_pixel:y2_pixel, x1_pixel] = 1
                # else:
                #     bev_feature_class_gt[y1_pixel:y2_pixel, x1_pixel:x2_pixel] = 1
            
            # kq_cross = F.normalize(torch.matmul(class_teacher_key, student_features_batch.reshape(channel, -1)), dim=0).cuda()
            if mode == 'gt':
                loss += torch.mean(torch.norm(class_teacher_key - class_key, p=2, dim=0)) / class_teacher_key.shape[0]
            elif mode == 'gt + attention':
                loss += torch.mean(torch.norm(class_teacher_key - class_key, p=2, dim=0)) / class_teacher_key.shape[0]
                kq_cross = torch.matmul(class_teacher_key, student_features_batch.reshape(channel, -1)).cuda()
                kq_cross = kq_cross.view(class_teacher_key.shape[0], height, width)
                kq_cross_mean = torch.mean(kq_cross, dim=0)
                #import pdb; pdb.set_trace()
                kqv_cross = kq_cross_mean[np.newaxis, :, :] * teacher_features_batch
                #import pdb; pdb.set_trace()
                
                # kq_self = F.normalize(torch.matmul(class_key, student_features_batch.reshape(channel, -1)), dim=0).cuda()
                kq_self = torch.matmul(class_key, student_features_batch.reshape(channel, -1)).cuda()
                kq_self = kq_self.view(class_key.shape[0], height, width)
                kq_self_mean = torch.mean(kq_self, dim=0)
                kqv_self = kq_self_mean[np.newaxis, :, :] * student_features_batch
                mse_loss = 0
                weight = 1
                squared_errors = ((kqv_cross - kqv_self)*weight)**2
                #squared_errors2 = ((kq_cross - kq_self)*weight)**2
                mse = torch.mean(squared_errors)
                #mse2 = torch.mean(squared_errors2)
                #mse2 /= class_key.shape[0]
                loss2 += mse
            else:
                kq_cross = torch.matmul(class_teacher_key, student_features_batch.reshape(channel, -1)).cuda()
                kq_cross = kq_cross.view(class_teacher_key.shape[0], height, width)
                kq_cross_mean = torch.mean(kq_cross, dim=0)
                #import pdb; pdb.set_trace()
                kqv_cross = kq_cross_mean[np.newaxis, :, :] * teacher_features_batch
                #import pdb; pdb.set_trace()
                
                # kq_self = F.normalize(torch.matmul(class_key, student_features_batch.reshape(channel, -1)), dim=0).cuda()
                kq_self = torch.matmul(class_key, student_features_batch.reshape(channel, -1)).cuda()
                kq_self = kq_self.view(class_key.shape[0], height, width)
                kq_self_mean = torch.mean(kq_self, dim=0)
                kqv_self = kq_self_mean[np.newaxis, :, :] * student_features_batch
                mse_loss = 0
                weight = 1
                squared_errors = ((kqv_cross - kqv_self)*weight)**2
                #squared_errors2 = ((kq_cross - kq_self)*weight)**2
                mse = torch.mean(squared_errors)
                #mse2 = torch.mean(squared_errors2)
                #mse2 /= class_key.shape[0]
                loss += mse
            #loss += mse2
            
            # for l in range(class_key.shape[0]):
            #     squared_errors = (bev_feature_class_gt - kq_cross[l])**2
            # # # 평균 제곱 오차 계산
            #     mse = torch.mean(squared_errors)
            #     mse_loss += mse
            # loss += (mse_loss / class_key.shape[0])
            
        #loss /= len(batch_class_features_dict[i].keys())
    if mode == 'gt + attention':
        loss /= batch_size
        loss2 /= batch_size
        return loss, loss2
    loss /= batch_size
    #pdb.set_trace()
    return loss


def cal_bev_attention_loss2_no_roi_add_self(batch_teacher, batch, index = None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...]. 292 * 1 * 512
        labels: ground truth of shape [bsz]. 292
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    loss = torch.tensor([0], dtype=torch.float32).cuda()
    
    teacher_features = batch_teacher['spatial_features_2d'].detach()
    # Extract features from the student model
    student_features = batch['spatial_features_2d']
    # student_features = batch['spatial_features_2d_group'][:, index]
    # Extract ground truth class information from the batch
    
    dataset_cfg = batch['dataset_cfg']
    min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
    min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
    max_x = dataset_cfg.POINT_CLOUD_RANGE[3]
    max_y = dataset_cfg.POINT_CLOUD_RANGE[4]
    voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
    voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
    
    # Extract region of interests (rois) from the batch
    rois = batch['gt_boxes']
    original_rois_shape2 = rois.shape
    original_rois_shape = (rois.shape[0] * rois.shape[1], rois.shape[2]) 
    selected_rois = rois[(rois[:, :, 0] > min_x) & (rois[:, :, 1] > min_y) & (rois[:, :, 0] < max_x) & (rois[:, :, 1] < max_y)]
    filled_rois = torch.zeros(original_rois_shape, dtype=rois.dtype, device=rois.device)
    if original_rois_shape != selected_rois.shape:
        filled_rois[:selected_rois.shape[0], :] = selected_rois
        rois = filled_rois.view(original_rois_shape2)
    # original_rois_shape과 동일한 크기의 텐서를 생성하고 0으로 초기화

    try:
        gt_class = rois[:, :, 7]
    except:
        gt_class = rois[:, 7]
        
    # Extract configuration information from the dataset_cfg in the batch
    # Get dimensions of the features
    batch_size, channel, height, width = teacher_features.size(0), teacher_features.size(1), teacher_features.size(2), teacher_features.size(3)
    x_down_sample_ratio = ((max_x - min_x) / voxel_size_x) / width
    y_down_sample_ratio = ((max_y - min_y) / voxel_size_y) / height
    # Get the number of rois    
    roi_size = rois.size(1)

    
    batch_class_features_dict = {} # [batch][class][i] = features, x1, x2, y1, y2
    batch_teacher_class_features_dict = {} # [batch][class][i] = features, x1, x2, y1, y2
    # Iterate over instances
    for i in range(batch_size):
        # Iterate over rois for the current instance
        student_features_batch = student_features[i]
        teacher_features_batch = teacher_features[i]
        for j in range(roi_size):
            # Get the current roi's class label
            class_label = int(gt_class[i, j].item())

            # Skip instances with class label 0
            if class_label != 1 and class_label != 2 and class_label != 3:
                continue

            # Calculate normalized coordinates of the current roi for the current instance
            x1 = (rois[i, j, 0] - rois[i, j, 3] / 2 - min_x) / (voxel_size_x * x_down_sample_ratio)
            x2 = (rois[i, j, 0] + rois[i, j, 3] / 2 - min_x) / (voxel_size_x * x_down_sample_ratio)
            y1 = (rois[i, j, 1] - rois[i, j, 4] / 2 - min_y) / (voxel_size_y * y_down_sample_ratio)
            y2 = (rois[i, j, 1] + rois[i, j, 4] / 2 - min_y) / (voxel_size_y * y_down_sample_ratio)

            # Convert normalized coordinates to pixel coordinates
            x1_pixel = (x1).clamp(0, width - 1).long()
            x2_pixel = (x2+1).clamp(0, width - 1).long()
            y1_pixel = (y1).clamp(0, height - 1).long()
            y2_pixel = (y2+1).clamp(0, height - 1).long()

            # Extract region feature map from student_features
            
            if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
                region_feature = student_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
                region_feature_teacher = teacher_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
            elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
                region_feature = student_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
                region_feature_teacher = teacher_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
            elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
                region_feature = student_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
                region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
            else:
                region_feature = student_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
                region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
            
            region_feature = region_feature.mean(dim=(1, 2))
            region_feature_teacher = region_feature_teacher.mean(dim=(1, 2))
            # Initialize list for the current roi if not present
            if  torch.isnan(region_feature).any().item():
                pdb.set_trace()
            
            if i not in batch_class_features_dict:
                batch_class_features_dict[i] = {}
                batch_teacher_class_features_dict[i] = {}
            if class_label not in batch_class_features_dict[i]:
                batch_class_features_dict[i][class_label] = []
                batch_teacher_class_features_dict[i][class_label] = []

            # Add the region feature map to the list for the current roi and its associated class label
            batch_class_features_dict[i][class_label].append([region_feature, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
            batch_teacher_class_features_dict[i][class_label].append([region_feature_teacher, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
            
        
        #batch = i
        if i not in batch_class_features_dict.keys() or i not in batch_teacher_class_features_dict.keys():
            continue
        
        for label in batch_class_features_dict[i].keys():
            bev_feature_class_gt = torch.zeros(height, width).cuda()
            class_key = torch.tensor([]).cuda()
            class_teacher_key = torch.tensor([]).cuda()
            
            for k in range(len(batch_class_features_dict[i][label])):
                class_key = torch.cat((class_key, batch_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
                class_teacher_key = torch.cat((class_teacher_key, batch_teacher_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
                
                # x1_pixel, x2_pixel, y1_pixel, y2_pixel = batch_class_features_dict[i][label][k][1:]
                
                # if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
                #     bev_feature_class_gt[y1_pixel, x1_pixel] = 1
                # elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
                #     bev_feature_class_gt[y1_pixel, x1_pixel:x2_pixel] = 1
                # elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
                #     bev_feature_class_gt[y1_pixel:y2_pixel, x1_pixel] = 1
                # else:
                #     bev_feature_class_gt[y1_pixel:y2_pixel, x1_pixel:x2_pixel] = 1
            
            # kq_cross = F.normalize(torch.matmul(class_teacher_key, student_features_batch.reshape(channel, -1)), dim=0).cuda()
            kq_cross = torch.matmul(class_teacher_key, teacher_features_batch.reshape(channel, -1)).cuda()
            kq_cross = kq_cross.view(class_teacher_key.shape[0], height, width)
            kq_cross_mean = torch.mean(kq_cross, dim=0)
            #import pdb; pdb.set_trace()
            kqv_cross = kq_cross_mean[np.newaxis, :, :] * teacher_features_batch
            #import pdb; pdb.set_trace()
            
            # kq_self = F.normalize(torch.matmul(class_key, student_features_batch.reshape(channel, -1)), dim=0).cuda()
            kq_self = torch.matmul(class_key, student_features_batch.reshape(channel, -1)).cuda()
            kq_self = kq_self.view(class_key.shape[0], height, width)
            kq_self_mean = torch.mean(kq_self, dim=0)
            kqv_self = kq_self_mean[np.newaxis, :, :] * student_features_batch
            mse_loss = 0
            weight = 1
            loss += torch.mean(torch.norm((kqv_cross - kqv_self), p=2, dim=0))
            # squared_errors = ((kqv_cross - kqv_self)*weight)**2
            # #squared_errors2 = ((kq_cross - kq_self)*weight)**2
            # mse = torch.mean(squared_errors)
            # #mse2 = torch.mean(squared_errors2)
            # #mse2 /= class_key.shape[0]
            # loss += mse
            #loss += mse2
            
            # for l in range(class_key.shape[0]):
            #     squared_errors = (bev_feature_class_gt - kq_cross[l])**2
            # # # 평균 제곱 오차 계산
            #     mse = torch.mean(squared_errors)
            #     mse_loss += mse
            # loss += (mse_loss / class_key.shape[0])
            
        #loss /= len(batch_class_features_dict[i].keys())
    
    #loss /= batch_size
    #pdb.set_trace()
    return loss


def cal_bev_attention_loss2_no_roi_add_norm_self(batch_teacher, batch, index = None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...]. 292 * 1 * 512
        labels: ground truth of shape [bsz]. 292
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    loss = torch.tensor([0], dtype=torch.float32).cuda()
    
    teacher_features = batch_teacher['spatial_features_2d'].detach()
    # Extract features from the student model
    student_features = batch['spatial_features_2d']
    # student_features = batch['spatial_features_2d_group'][:, index]
    # Extract ground truth class information from the batch
    
    dataset_cfg = batch['dataset_cfg']
    min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
    min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
    max_x = dataset_cfg.POINT_CLOUD_RANGE[3]
    max_y = dataset_cfg.POINT_CLOUD_RANGE[4]
    voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
    voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
    
    # Extract region of interests (rois) from the batch
    rois = batch['gt_boxes']
    original_rois_shape2 = rois.shape
    original_rois_shape = (rois.shape[0] * rois.shape[1], rois.shape[2]) 
    selected_rois = rois[(rois[:, :, 0] > min_x) & (rois[:, :, 1] > min_y) & (rois[:, :, 0] < max_x) & (rois[:, :, 1] < max_y)]
    filled_rois = torch.zeros(original_rois_shape, dtype=rois.dtype, device=rois.device)
    if original_rois_shape != selected_rois.shape:
        filled_rois[:selected_rois.shape[0], :] = selected_rois
        rois = filled_rois.view(original_rois_shape2)
    # original_rois_shape과 동일한 크기의 텐서를 생성하고 0으로 초기화

    try:
        gt_class = rois[:, :, 7]
    except:
        gt_class = rois[:, 7]
        
    # Extract configuration information from the dataset_cfg in the batch
    # Get dimensions of the features
    batch_size, channel, height, width = teacher_features.size(0), teacher_features.size(1), teacher_features.size(2), teacher_features.size(3)
    x_down_sample_ratio = ((max_x - min_x) / voxel_size_x) / width
    y_down_sample_ratio = ((max_y - min_y) / voxel_size_y) / height
    # Get the number of rois    
    roi_size = rois.size(1)

    
    batch_class_features_dict = {} # [batch][class][i] = features, x1, x2, y1, y2
    batch_teacher_class_features_dict = {} # [batch][class][i] = features, x1, x2, y1, y2
    # Iterate over instances
    for i in range(batch_size):
        # Iterate over rois for the current instance
        student_features_batch = student_features[i]
        teacher_features_batch = teacher_features[i]
        for j in range(roi_size):
            # Get the current roi's class label
            class_label = int(gt_class[i, j].item())

            # Skip instances with class label 0
            if class_label != 1 and class_label != 2 and class_label != 3:
                continue

            # Calculate normalized coordinates of the current roi for the current instance
            x1 = (rois[i, j, 0] - rois[i, j, 3] / 2 - min_x) / (voxel_size_x * x_down_sample_ratio)
            x2 = (rois[i, j, 0] + rois[i, j, 3] / 2 - min_x) / (voxel_size_x * x_down_sample_ratio)
            y1 = (rois[i, j, 1] - rois[i, j, 4] / 2 - min_y) / (voxel_size_y * y_down_sample_ratio)
            y2 = (rois[i, j, 1] + rois[i, j, 4] / 2 - min_y) / (voxel_size_y * y_down_sample_ratio)

            # Convert normalized coordinates to pixel coordinates
            x1_pixel = (x1).clamp(0, width - 1).long()
            x2_pixel = (x2+1).clamp(0, width - 1).long()
            y1_pixel = (y1).clamp(0, height - 1).long()
            y2_pixel = (y2+1).clamp(0, height - 1).long()

            # Extract region feature map from student_features
            
            if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
                region_feature = student_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
                region_feature_teacher = teacher_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
            elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
                region_feature = student_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
                region_feature_teacher = teacher_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
            elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
                region_feature = student_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
                region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
            else:
                region_feature = student_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
                region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
            
            region_feature = region_feature.mean(dim=(1, 2))
            region_feature_teacher = region_feature_teacher.mean(dim=(1, 2))
            # Initialize list for the current roi if not present
            if  torch.isnan(region_feature).any().item():
                pdb.set_trace()
            
            if i not in batch_class_features_dict:
                batch_class_features_dict[i] = {}
                batch_teacher_class_features_dict[i] = {}
            if class_label not in batch_class_features_dict[i]:
                batch_class_features_dict[i][class_label] = []
                batch_teacher_class_features_dict[i][class_label] = []

            # Add the region feature map to the list for the current roi and its associated class label
            batch_class_features_dict[i][class_label].append([region_feature, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
            batch_teacher_class_features_dict[i][class_label].append([region_feature_teacher, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
            
        
        #batch = i
        if i not in batch_class_features_dict.keys() or i not in batch_teacher_class_features_dict.keys():
            continue
        
        for label in batch_class_features_dict[i].keys():
            bev_feature_class_gt = torch.zeros(height, width).cuda()
            class_key = torch.tensor([]).cuda()
            class_teacher_key = torch.tensor([]).cuda()
            
            for k in range(len(batch_class_features_dict[i][label])):
                class_key = torch.cat((class_key, batch_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
                class_teacher_key = torch.cat((class_teacher_key, batch_teacher_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
                
                # x1_pixel, x2_pixel, y1_pixel, y2_pixel = batch_class_features_dict[i][label][k][1:]
                
                # if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
                #     bev_feature_class_gt[y1_pixel, x1_pixel] = 1
                # elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
                #     bev_feature_class_gt[y1_pixel, x1_pixel:x2_pixel] = 1
                # elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
                #     bev_feature_class_gt[y1_pixel:y2_pixel, x1_pixel] = 1
                # else:
                #     bev_feature_class_gt[y1_pixel:y2_pixel, x1_pixel:x2_pixel] = 1
            
            # kq_cross = F.normalize(torch.matmul(class_teacher_key, student_features_batch.reshape(channel, -1)), dim=0).cuda()
            kq_cross = torch.matmul(class_teacher_key, teacher_features_batch.reshape(channel, -1)).cuda()
            kq_cross = kq_cross.view(class_teacher_key.shape[0], height, width)
            kq_cross_mean = torch.mean(kq_cross, dim=0)
            #import pdb; pdb.set_trace()
            kqv_cross = kq_cross_mean[np.newaxis, :, :] * teacher_features_batch
            #import pdb; pdb.set_trace()
            
            # kq_self = F.normalize(torch.matmul(class_key, student_features_batch.reshape(channel, -1)), dim=0).cuda()
            kq_self = torch.matmul(class_key, student_features_batch.reshape(channel, -1)).cuda()
            kq_self = kq_self.view(class_key.shape[0], height, width)
            kq_self_mean = torch.mean(kq_self, dim=0)
            kqv_self = kq_self_mean[np.newaxis, :, :] * student_features_batch
            mse_loss = 0
            weight = 1
            #squared_errors = ((kqv_cross - kqv_self)*weight)**2
            loss += torch.mean(torch.norm((kqv_cross - kqv_self), p=2, dim=0))
            # squared_errors = ((kqv_cross - kqv_self)*weight)**2
            # #squared_errors2 = ((kq_cross - kq_self)*weight)**2
            # mse = torch.mean(squared_errors)
            # #mse2 = torch.mean(squared_errors2)
            # #mse2 /= class_key.shape[0]
            # loss += mse
            #loss += mse2
            
            # for l in range(class_key.shape[0]):
            #     squared_errors = (bev_feature_class_gt - kq_cross[l])**2
            # # # 평균 제곱 오차 계산
            #     mse = torch.mean(squared_errors)
            #     mse_loss += mse
            # loss += (mse_loss / class_key.shape[0])
            
        #loss /= len(batch_class_features_dict[i].keys())
    
    loss /= batch_size
    #pdb.set_trace()
    return loss



def cal_bev_attention_loss2_no_roi_add_mean_clamp_norm_self(batch_teacher, batch, index = None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...]. 292 * 1 * 512
        labels: ground truth of shape [bsz]. 292
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    loss = torch.tensor([0], dtype=torch.float32).cuda()
    
    teacher_features = batch_teacher['spatial_features_2d'].detach()
    # Extract features from the student model
    student_features = batch['spatial_features_2d']
    # student_features = batch['spatial_features_2d_group'][:, index]
    # Extract ground truth class information from the batch
    
    dataset_cfg = batch['dataset_cfg']
    min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
    min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
    max_x = dataset_cfg.POINT_CLOUD_RANGE[3]
    max_y = dataset_cfg.POINT_CLOUD_RANGE[4]
    voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
    voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
    
    # Extract region of interests (rois) from the batch
    rois = batch['gt_boxes']
    original_rois_shape2 = rois.shape
    original_rois_shape = (rois.shape[0] * rois.shape[1], rois.shape[2]) 
    selected_rois = rois[(rois[:, :, 0] > min_x) & (rois[:, :, 1] > min_y) & (rois[:, :, 0] < max_x) & (rois[:, :, 1] < max_y)]
    filled_rois = torch.zeros(original_rois_shape, dtype=rois.dtype, device=rois.device)
    if original_rois_shape != selected_rois.shape:
        filled_rois[:selected_rois.shape[0], :] = selected_rois
        rois = filled_rois.view(original_rois_shape2)
    # original_rois_shape과 동일한 크기의 텐서를 생성하고 0으로 초기화

    try:
        gt_class = rois[:, :, 7]
    except:
        gt_class = rois[:, 7]
        
    # Extract configuration information from the dataset_cfg in the batch
    # Get dimensions of the features
    batch_size, channel, height, width = teacher_features.size(0), teacher_features.size(1), teacher_features.size(2), teacher_features.size(3)
    x_down_sample_ratio = ((max_x - min_x) / voxel_size_x) / width
    y_down_sample_ratio = ((max_y - min_y) / voxel_size_y) / height
    # Get the number of rois    
    roi_size = rois.size(1)

    
    batch_class_features_dict = {} # [batch][class][i] = features, x1, x2, y1, y2
    batch_teacher_class_features_dict = {} # [batch][class][i] = features, x1, x2, y1, y2
    # Iterate over instances
    for i in range(batch_size):
        # Iterate over rois for the current instance
        student_features_batch = student_features[i]
        teacher_features_batch = teacher_features[i]
        for j in range(roi_size):
            # Get the current roi's class label
            class_label = int(gt_class[i, j].item())

            # Skip instances with class label 0
            if class_label != 1 and class_label != 2 and class_label != 3:
                continue

            # Calculate normalized coordinates of the current roi for the current instance
            x1 = (rois[i, j, 0] - rois[i, j, 3] / 2 - min_x) / (voxel_size_x * x_down_sample_ratio)
            x2 = (rois[i, j, 0] + rois[i, j, 3] / 2 - min_x) / (voxel_size_x * x_down_sample_ratio)
            y1 = (rois[i, j, 1] - rois[i, j, 4] / 2 - min_y) / (voxel_size_y * y_down_sample_ratio)
            y2 = (rois[i, j, 1] + rois[i, j, 4] / 2 - min_y) / (voxel_size_y * y_down_sample_ratio)

            # Convert normalized coordinates to pixel coordinates
            x_pixel = ((x1+x2) / 2).clamp(0, width - 1).long()
            y_pixel = ((y1+y2) / 2).clamp(0, height - 1).long()
            # x1_pixel = (x1).clamp(0, width - 1).long()
            # x2_pixel = (x2+1).clamp(0, width - 1).long()
            # y1_pixel = (y1).clamp(0, height - 1).long()
            # y2_pixel = (y2+1).clamp(0, height - 1).long()

            # Extract region feature map from student_features
            
            # if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
            #     region_feature = student_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
            #     region_feature_teacher = teacher_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
            # elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
            #     region_feature = student_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
            #     region_feature_teacher = teacher_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
            # elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
            #     region_feature = student_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
            #     region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
            # else:
            #     region_feature = student_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
            #     region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
            
            #torch.norm(tensor_512_M_N, p=2, dim=(1, 2))
            #region_feature = torch.norm(region_feature, p=2, dim=(1,2))
            #region_feature_teacher = torch.norm(region_feature_teacher, p=2, dim=(1,2))
            region_feature = student_features[i, :, y_pixel, x_pixel]
            region_feature_teacher = teacher_features[i, :, y_pixel, x_pixel]
            
            # region_feature = region_feature.mean(dim=(1, 2))
            # region_feature_teacher = region_feature_teacher.mean(dim=(1, 2))
            # Initialize list for the current roi if not present
            if  torch.isnan(region_feature).any().item():
                pdb.set_trace()
            
            if i not in batch_class_features_dict:
                batch_class_features_dict[i] = {}
                batch_teacher_class_features_dict[i] = {}
            if class_label not in batch_class_features_dict[i]:
                batch_class_features_dict[i][class_label] = []
                batch_teacher_class_features_dict[i][class_label] = []
            x1_pixel = 0
            x2_pixel = 0
            y1_pixel = 0
            y2_pixel = 0
            # Add the region feature map to the list for the current roi and its associated class label
            batch_class_features_dict[i][class_label].append([region_feature, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
            batch_teacher_class_features_dict[i][class_label].append([region_feature_teacher, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
            
        
        #batch = i
        if i not in batch_class_features_dict.keys() or i not in batch_teacher_class_features_dict.keys():
            continue
        
        for label in batch_class_features_dict[i].keys():
            bev_feature_class_gt = torch.zeros(height, width).cuda()
            class_key = torch.tensor([]).cuda()
            class_teacher_key = torch.tensor([]).cuda()
            
            for k in range(len(batch_class_features_dict[i][label])):
                class_key = torch.cat((class_key, batch_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
                class_teacher_key = torch.cat((class_teacher_key, batch_teacher_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
                
                # x1_pixel, x2_pixel, y1_pixel, y2_pixel = batch_class_features_dict[i][label][k][1:]
                
                # if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
                #     bev_feature_class_gt[y1_pixel, x1_pixel] = 1
                # elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
                #     bev_feature_class_gt[y1_pixel, x1_pixel:x2_pixel] = 1
                # elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
                #     bev_feature_class_gt[y1_pixel:y2_pixel, x1_pixel] = 1
                # else:
                #     bev_feature_class_gt[y1_pixel:y2_pixel, x1_pixel:x2_pixel] = 1
            
            # kq_cross = F.normalize(torch.matmul(class_teacher_key, student_features_batch.reshape(channel, -1)), dim=0).cuda()
            kq_cross = torch.matmul(class_teacher_key, teacher_features_batch.reshape(channel, -1)).cuda()
            kq_cross = kq_cross.view(class_teacher_key.shape[0], height, width)
            kq_cross_mean = torch.mean(kq_cross, dim=0)
            #import pdb; pdb.set_trace()
            kqv_cross = kq_cross_mean[np.newaxis, :, :] * teacher_features_batch
            #import pdb; pdb.set_trace()
            
            # kq_self = F.normalize(torch.matmul(class_key, student_features_batch.reshape(channel, -1)), dim=0).cuda()
            kq_self = torch.matmul(class_key, student_features_batch.reshape(channel, -1)).cuda()
            kq_self = kq_self.view(class_key.shape[0], height, width)
            kq_self_mean = torch.mean(kq_self, dim=0)
            kqv_self = kq_self_mean[np.newaxis, :, :] * student_features_batch
            mse_loss = 0
            weight = 1
            #squared_errors = ((kqv_cross - kqv_self)*weight)**2
            loss += torch.mean(torch.norm((kqv_cross - kqv_self), p=2, dim=0))
            # squared_errors = ((kqv_cross - kqv_self)*weight)**2
            # #squared_errors2 = ((kq_cross - kq_self)*weight)**2
            # mse = torch.mean(squared_errors)
            # #mse2 = torch.mean(squared_errors2)
            # #mse2 /= class_key.shape[0]
            # loss += mse
            #loss += mse2
            
            # for l in range(class_key.shape[0]):
            #     squared_errors = (bev_feature_class_gt - kq_cross[l])**2
            # # # 평균 제곱 오차 계산
            #     mse = torch.mean(squared_errors)
            #     mse_loss += mse
            # loss += (mse_loss / class_key.shape[0])
            
        #loss /= len(batch_class_features_dict[i].keys())
    
    loss /= batch_size
    #pdb.set_trace()
    return loss
def make_pseudo_label(tb_dict_teacher, confidence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = len(tb_dict_teacher)

    # Find the maximum number of pred_boxes in a batch
    max_num_boxes = max(len(tb_dict_teacher[bc_size]['pred_boxes']) for bc_size in range(batch))

    # Initialize lists to store pred_boxes, pred_scores, and pred_labels
    all_pred_boxes = []

    # Loop over batches
    for bc_size in range(batch):
        pred_boxes = tb_dict_teacher[bc_size]['pred_boxes'].to(device).detach()
        pred_scores = tb_dict_teacher[bc_size]['pred_scores'].to(device).detach()
        pred_labels = tb_dict_teacher[bc_size]['pred_labels'].to(device).detach()

        # Concatenate pred_labels to pred_boxes
        pred_boxes_with_labels = torch.cat((pred_boxes, pred_labels.view(-1, 1)), dim=1)

        # Remove pred_boxes with pred_scores less than confidence
        mask = pred_scores >= confidence
        pred_boxes_with_labels = pred_boxes_with_labels[mask]

        # Pad pred_boxes if the number is less than max_num_boxes
        num_padding = max_num_boxes - len(pred_boxes_with_labels)
        if num_padding > 0:
            padding = torch.zeros((num_padding, pred_boxes_with_labels.shape[1]), device=device)
            pred_boxes_with_labels = torch.cat((pred_boxes_with_labels, padding), dim=0)

        # Append to the lists
        all_pred_boxes.append(pred_boxes_with_labels[:, :])

    # Concatenate the lists to create the final tensor
    final_pred_boxes = torch.stack(all_pred_boxes, dim=0)

    return final_pred_boxes
    



def train_one_epoch(model, model_teacher, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, mimic_weight=1.0, mimic_mode='roi', supervised=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):

        try:
            #import pdb; pdb.set_trace()
            batch, batch_teacher = next(dataloader_iter)
            #batch = next(dataloader_iter)

        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch, batch_teacher = next(dataloader_iter)

            print('new iters')

        #batch_teacher = batch.copy()
        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        model_teacher.eval()
        optimizer.zero_grad()

        load_data_to_gpu(batch_teacher)
        batch_teacher['mimic'] = 'mimic'
        batch['mimic'] = 'mimic'
        with torch.no_grad():
            #temp1, batch_teacher_new = model_func(model_teacher, batch_teacher)
            # teacher: 3D backbone -> map2BEV
            tb_dict_teacher, disp_dict_teacher, batch_teacher_new = model_teacher(batch_teacher)
            batch['gt_boxes'] = make_pseudo_label(tb_dict_teacher, 0.3)

        
        # roi_head_iou_layers_state_dict = model_batch.roi_head.iou_layers.state_dict()

        # # # Load the state_dict into model.roi_head_iou_layers
        # model.roi_head.iou_layers.load_state_dict(roi_head_iou_layers_state_dict)
        
        
        # batch['rois_mimic'] = batch_teacher_new['rois_mimic'].clone()
        #TODO batch['gt_boxes'].shape = (12, 132, 8) -> numpy
        #tb_dict_teacher[batch]['pred_boxes'].shape = (167, 7)
        #tb_dict_teacher[batch]['pred_scores'].shape = (167)
        #tb_dict_teacher[batch]['pred_labels'].shape = (167)
        #tb_dict_teacher[0].keys() -> dict_keys(['pred_boxes', 'pred_scores', 'pred_labels'])

        temp, batch_new = model_func(model, batch)
        loss, tb_dict, disp_dict = temp
        #import pdb; pdb.set_trace()
        if supervised:
            loss_sum = loss
            #print(1)
        else:
            #loss_mimic = cal_mimic_loss(batch_teacher_new, batch_new, 'all')
            #loss_mimic2 = cal_mimic_loss(batch_teacher_new, batch_new, 'gt') * 0.2
            #loss_sum = loss + loss_mimic
            #loss_mimic2, loss_attention2_no_roi = cal_bev_attention_loss2_no_roi_add(batch_teacher_new, batch_new, mode='gt + attention')
            loss_mimic2 = cal_bev_attention_loss2_no_roi_add(batch_teacher_new, batch_new, mode='gt')
            #loss_attention2_no_roi = cal_bev_attention_loss2_no_roi_add(batch_teacher_new, batch_new)
            #loss_sum = loss + loss_attention2_no_roi + loss_mimic2
            loss_sum = loss + loss_mimic2
            #print(2)

        #loss_linear_probing = cal_linear_probing(batch_teacher_new, batch_new)
        #loss_mimic = cal_mimic_loss(batch_teacher_new, batch_new, model.module.model_cfg.ROI_HEAD, 'roi')
        #loss_attention2_no_roi = 0
        # for i in range(4):
        #loss_attention2_no_roi = cal_bev_attention_loss2_no_roi_add(batch_teacher_new, batch_new)
        #loss_attention2_no_roi = cal_bev_attention_loss2_no_roi(batch_teacher_new, batch_new)
        #loss_attention2_no_roi = cal_bev_attention_loss2_no_roi_add_softmax(batch_teacher_new, batch_new)
        #features, labels, mask = cal_inter_class_loss(batch_new, model.module.model_cfg.ROI_HEAD)
        
        #loss_inter_class = cal_inter_class_loss(batch_new, model.module.model_cfg.ROI_HEAD, 0.07, 'all', 0.07)
        #loss_sum = loss + loss_mimic * mimic_weight
        
        #loss_sum = loss #+ loss_attention2_no_roi
       
        #loss_sum = loss + loss_linear_probing * mimic_weight
        #loss_sum = loss_inter_class * mimic_weight
        loss_sum.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        if supervised:
            disp_dict.update({'loss': loss.item(),  'lr': cur_lr})
        else:
            disp_dict.update({'loss': loss.item(), 'loss_mimic': loss_mimic2.item(), 'lr': cur_lr})
            #disp_dict.update({'loss': loss.item(), 'loss_attention': loss_attention2_no_roi.item(), 'loss_mimic2': loss_mimic2.item(), 'lr': cur_lr})
        #disp_dict.update({'loss': loss.item(), 'loss_inter_class': loss_inter_class.item(), 'lr': cur_lr})
        #disp_dict.update({'loss': loss.item(), 'loss_mimic': loss_mimic.item(), 'lr': cur_lr})
        #disp_dict.update({'loss': loss.item(), 'loss_attention': loss_attention2_no_roi.item(), 'lr': cur_lr})
        
        

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model_mimic(model, model_teacher, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                source_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None, mimic_weight=1, mimic_mode='roi', supervised=None, EMA=0):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:

            # with torch.no_grad():
            #     teacher_params = model_teacher.state_dict()
            #     model_params = model.state_dict()
            #     # # # Load the state_dict into model.roi_head_iou_layers
            #     #model.roi_head.iou_layers.load_state_dict(roi_head_iou_layers_state_dict)
            #     for key in model_params.keys():
            #         teacher_params[key] = model_params[key]
        
            #     model_teacher.load_state_dict(teacher_params)


            if EMA != 0:
                if cur_epoch % 3 ==0:
                    teacher_params = model_teacher.state_dict()
                    model_params = model.state_dict()
                    alpha = EMA
                    # # # Load the state_dict into model.roi_head_iou_layers
                    #model.roi_head.iou_layers.load_state_dict(roi_head_iou_layers_state_dict)
                    for key in teacher_params.keys():
                        teacher_params[key] = alpha * teacher_params[key] + (1 - alpha) * model_params[key]
                    
                    model_teacher.load_state_dict(teacher_params)

            if source_sampler is not None:
                source_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, model_teacher, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                mimic_weight=mimic_weight,
                mimic_mode=mimic_mode,
                supervised=supervised
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
