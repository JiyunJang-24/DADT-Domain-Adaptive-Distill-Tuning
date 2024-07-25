from .detector3d_template import Detector3DTemplate
from .detector3d_template_multi_db import Detector3DTemplate_M_DB
from pcdet.utils import common_utils
import numpy as np
import torch

class ADPT(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def get_attentionmap(self, batch_dict, model_cfg): 
        teacher_features = batch_dict['spatial_features_2d'].detach()
        rois = batch_dict['gt_boxes']
        # teacher_features = batch_teacher['spatial_features_2d'].detach()
        # # Extract features from the student model
        # student_features = batch['spatial_features_2d']
        # dataset_cfg = batch_dict['dataset_cfg']
        # min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
        # min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
        # max_x = dataset_cfg.POINT_CLOUD_RANGE[3]
        # max_y = dataset_cfg.POINT_CLOUD_RANGE[4]
        # voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
        # voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
        np.save('pts', batch_dict['points'].cpu().numpy())
        np.save('gt_boxes', batch_dict['gt_boxes'].cpu().numpy())
        ## kitti
        min_x = -75.2
        min_y = -75.2
        max_x = 75.2
        max_y = 75.2
        down_sample_ratio = 8
        voxel_size_x = 0.1
        voxel_size_y = 0.1
        mode = 'gt + attention'
        # Extract region of interests (rois) from the batch
        # rois = batch['gt_boxes']
     
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
            # student_features_batch = student_features[i]
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
                    # region_feature = student_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
                    region_feature_teacher = teacher_features[i, :, y1_pixel, x2_pixel].unsqueeze(1).unsqueeze(1)
                elif x1_pixel != x2_pixel and y1_pixel == y2_pixel:
                    # region_feature = student_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
                    region_feature_teacher = teacher_features[i, :, y1_pixel, x1_pixel:x2_pixel].unsqueeze(1)
                elif x1_pixel == x2_pixel and y1_pixel != y2_pixel:
                    # region_feature = student_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
                    region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x2_pixel].unsqueeze(2)
                else:
                    # region_feature = student_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
                    region_feature_teacher = teacher_features[i, :, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
                
                # region_feature = region_feature.mean(dim=(1, 2))
                region_feature_teacher = region_feature_teacher.mean(dim=(1, 2))
                # Initialize list for the current roi if not present
                if  torch.isnan(region_feature_teacher).any().item():
                    pdb.set_trace()
                # import pdb; pdb.set_trace()
                if i not in batch_teacher_class_features_dict:
                    batch_teacher_class_features_dict[i] = {}
                if class_label not in batch_teacher_class_features_dict[i]:
                    # batch_class_features_dict[i][class_label] = []
                    batch_teacher_class_features_dict[i][class_label] = []

                # Add the region feature map to the list for the current roi and its associated class label
                # batch_class_features_dict[i][class_label].append([region_feature, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
                batch_teacher_class_features_dict[i][class_label].append([region_feature_teacher, x1_pixel, x2_pixel, y1_pixel, y2_pixel])
                
            
            #batch = i
            # if i not in batch_class_features_dict.keys() or i not in batch_teacher_class_features_dict.keys():
            if i not in batch_teacher_class_features_dict.keys():

                continue
            ## 같은 class label끼리 묶음
            for label in batch_teacher_class_features_dict[i].keys():
                bev_feature_class_gt = torch.zeros(height, width).cuda()
                class_key = torch.tensor([]).cuda()
                class_teacher_key = torch.tensor([]).cuda()
                
                for k in range(len(batch_teacher_class_features_dict[i][label])):
                    # class_key = torch.cat((class_key, batch_class_features_dict[i][label][k][0].unsqueeze(0)), dim=0)
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
                    # loss += torch.mean(torch.norm(class_teacher_key - class_key, p=2, dim=0)) / class_teacher_key.shape[0]
                    kq_cross = torch.matmul(class_teacher_key, teacher_features_batch.reshape(channel, -1)).cuda()
                    kq_cross = kq_cross.view(class_teacher_key.shape[0], height, width)
                    np.save('ours_feat2', kq_cross.cpu().numpy())

                    
                    # save bbox
                    boxes = []
                    for j in range(len(batch_teacher_class_features_dict[i][class_label])):
                        box = []
                        bboxes = batch_teacher_class_features_dict[i][class_label][j][1:]
                        for num in bboxes: 
                            num = num.cpu().item()
                            box.append(num)
                        boxes.append(np.array(box))
                    
                    np.save('ours_boxes2', np.array(boxes))
                    import pdb; pdb.set_trace()

    def roi(self, batch_dict, model_cfg): 
        
        features = batch_dict['spatial_features_2d'].detach()
        rois = batch_dict['gt_boxes']
        # once dataset cfg
        # min_x = -75.2
        # min_y = -75.2
        # max_x = 75.2 
        # max_y = 75.2
        # voxel_size_x = 0.1
        # voxel_size_y = 0.1
        # down_sample_ratio = 8

        # Kitti dataset cfg
        min_x = 0
        min_y = -40
        max_x = 70.4
        max_y = 40
        down_sample_ratio = 8
        voxel_size_x = 0.05
        voxel_size_y = 0.05

        batch_size, channels, height, width = features.shape
        down_sample_ratio = ((max_x - min_x) / voxel_size_x) / width
        # y_down_sample_ratio = ((max_y - min_y) / voxel_size_y) / height
        roi_size = rois.size(1)

        x1 = (rois[:, :, 0] - rois[:, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
        x2 = (rois[:, :, 0] + rois[:, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
        y1 = (rois[:, :, 1] - rois[:, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
        y2 = (rois[:, :, 1] + rois[:, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)

        mask = torch.zeros(batch_size, roi_size, height, width).bool().cuda()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        grid_y = grid_y[None, None].repeat(batch_size, roi_size, 1, 1).cuda()
        grid_x = grid_x[None, None].repeat(batch_size, roi_size, 1, 1).cuda()

        mask_y = (grid_y >= y1[:, :, None, None]) * (grid_y <= y2[:, :, None, None])
        mask_x = (grid_x >= x1[:, :, None, None]) * (grid_x <= x2[:, :, None, None])
        mask = (mask_y * mask_x).float()
        roi_features = []
        for k in range(roi_size): 
            _, _, i, j = torch.where(mask[:, k, :, :].unsqueeze(1))
            feature = features[:, :, i, j] 
            roi_features.append(feature)
        if len(roi_features) == 0: 
            return roi_features
        
        roi_features  = torch.cat(roi_features, dim = -1).permute(0, 2, 1)
        B, N, C = roi_features.shape
        roi_features = roi_features.contiguous().view(B*N, C)
        # np.save('20per',batch_dict['spatial_features_2d'].cpu().numpy())
        return roi_features

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        if self.unetscn:
            batch_dict = self.unetscn(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        
     
        # rois = self.roi(batch_dict, self.model_cfg)
        rois = self.get_attentionmap(batch_dict, self.model_cfg)

        # if self.training:
        #     loss, tb_dict, disp_dict = self.get_training_loss()

        #     ret_dict = {
        #         'loss': loss
        #     }
        #     return ret_dict, tb_dict, disp_dict
        # else:
        #     pred_dicts, recall_dicts = self.post_processing(batch_dict)
        return rois

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

