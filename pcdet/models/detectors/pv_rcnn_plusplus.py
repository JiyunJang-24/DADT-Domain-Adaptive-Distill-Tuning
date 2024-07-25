from .detector3d_template import Detector3DTemplate
from .detector3d_template_multi_db import Detector3DTemplate_M_DB
from pcdet.utils import common_utils
import numpy as np
import torch

class PVRCNNPlusPlus(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    
    def merge_double_flip(self, batch_dict):

        batch_size = batch_dict['batch_size']
        voxel_coords_ = []
        voxel_num_points_ = []
        voxels_ = []
        voxel_features_ = []
        indices3_ = []
        indices4_ = []
        features3_ = []
        features4_ = []

        for i in range(batch_size): 
            if i % 4 == 0:
                idxs = torch.where(batch_dict['voxel_coords'][:, 0] == i)[0]
                voxel_coords_.append(batch_dict['voxel_coords'][idxs, :])
                voxels_.append(batch_dict['voxels'][idxs, :, :])
                voxel_num_points_.append(batch_dict['voxel_num_points'][idxs])
                voxel_features_.append(batch_dict['voxel_features'][idxs])

                indices3 = batch_dict['multi_scale_3d_features']['x_conv3'].indices
                indices4 = batch_dict['multi_scale_3d_features']['x_conv4'].indices
                f3 = batch_dict['multi_scale_3d_features']['x_conv3'].features
                f4 = batch_dict['multi_scale_3d_features']['x_conv4'].features
                idxs3 = torch.where(indices3 == i)[0]
                idxs4 = torch.where(indices4 == i)[0]
                indices3_.append(indices3[idxs3,:])
                indices4_.append(indices4[idxs4,:])
                features3_.append(f3[idxs3, :])
                features4_.append(f4[idxs4, :])

       
        batch_dict['voxel_coords'] = torch.cat(voxel_coords_)
        batch_dict['voxel_num_points'] = torch.cat(voxel_num_points_)
        batch_dict['voxels'] = torch.cat(voxels_)
        batch_dict['voxel_features'] = torch.cat(voxel_features_)
        batch_dict['multi_scale_3d_features']['x_conv3'].indices = torch.cat(indices3_)
        batch_dict['multi_scale_3d_features']['x_conv4'].indices = torch.cat(indices4_)

        batch_dict['multi_scale_3d_features']['x_conv3'] = batch_dict['multi_scale_3d_features']['x_conv3'].replace_feature(torch.cat(features3_))
        batch_dict['multi_scale_3d_features']['x_conv4'] = batch_dict['multi_scale_3d_features']['x_conv4'].replace_feature(torch.cat(features4_))
                
        features = batch_dict['spatial_features_2d'] 
        sp_features = batch_dict['spatial_features'] 

        _, C, H, W = features.shape
        group_tensor = features.reshape(-1, 4, C, H, W)
        group_tensor[:, 1] = torch.flip(group_tensor[:, 1], dims=[2])
        group_tensor[:, 2] = torch.flip(group_tensor[:, 2], dims=[3])
        group_tensor[:, 3] = torch.flip(group_tensor[:, 3], dims=[2, 3])
        
        _, C, H, W = sp_features.shape
        sp_features = sp_features.reshape(-1, 4, C, H, W)[:, 0]

        batch_dict['spatial_features_2d'] = group_tensor[:, 0]
        batch_dict['spatial_features_2d_group'] = group_tensor
        batch_dict['spatial_features'] = sp_features
        
        return batch_dict 

    def forward(self, batch_dict):
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        batch_dict = self.vfe(batch_dict)
        if self.unetscn:
            batch_dict = self.unetscn(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        if self.training and self.is_distill: 
            assert batch_dict["batch_size"] % 4 == 0
            batch_dict = self.merge_double_flip(batch_dict)
            batch_dict['batch_size'] = batch_dict['batch_size'] // 4 

        batch_dict = self.dense_head(batch_dict)
        #import pdb; pdb.set_trace()
        batch_dict = self.roi_head.proposal_layer(
            batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.roi_head.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_targets_dict'] = targets_dict
            num_rois_per_scene = targets_dict['rois'].shape[1]
            if 'roi_valid_num' in batch_dict:
                batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]
        
        batch_dict = self.pfe(batch_dict)
        batch_dict = self.point_head(batch_dict)
        batch_dict = self.roi_head(batch_dict)
        
        #import pdb; pdb.set_trace()
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            

            if 'mimic' not in batch_dict.keys():
                return ret_dict, tb_dict, disp_dict
            else:
                return ret_dict, tb_dict, disp_dict, batch_dict
            
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if 'mimic' not in batch_dict.keys():
                return pred_dicts, recall_dicts
            else:
                return pred_dicts, recall_dicts, batch_dict
            #return pred_dicts, recall_dicts

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

class SemiPVRCNNPlusPlus(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_type = None

    def set_model_type(self, model_type):
        assert model_type in ['origin', 'teacher', 'student']
        self.model_type = model_type
        self.dense_head.model_type = model_type
        self.point_head.model_type = model_type
        self.roi_head.model_type = model_type

    def forward(self, batch_dict):
        if self.model_type == 'origin':
            batch_dict = self.vfe(batch_dict)
            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.map_to_bev_module(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)

            batch_dict = self.roi_head.proposal_layer(
                batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )
            if self.training:
                targets_dict = self.roi_head.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
                batch_dict['roi_targets_dict'] = targets_dict
                num_rois_per_scene = targets_dict['rois'].shape[1]
                if 'roi_valid_num' in batch_dict:
                    batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

            batch_dict = self.pfe(batch_dict)
            batch_dict = self.point_head(batch_dict)
            batch_dict = self.roi_head(batch_dict)

            if self.training:
                loss, tb_dict, disp_dict = self.get_training_loss()

                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts

        # teacher: (testing, return raw boxes)
        elif self.model_type == 'teacher':
            batch_dict = self.vfe(batch_dict)
            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.map_to_bev_module(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)

            batch_dict = self.roi_head.proposal_layer(
                    batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TEST']
                )

            batch_dict = self.pfe(batch_dict)
            batch_dict = self.point_head(batch_dict)
            batch_dict = self.roi_head(batch_dict)

            return batch_dict
        
        # student: (training, return (loss & raw boxes w/ gt_boxes) or raw boxes (w/o gt_boxes) for consistency)
        #          (testing, return final_boxes)
        elif self.model_type == 'student':
            batch_dict = self.vfe(batch_dict)
            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.map_to_bev_module(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)

            if 'gt_boxes' in batch_dict: 
                batch_dict = self.roi_head.proposal_layer(
                    batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
                )
            else:
                batch_dict = self.roi_head.proposal_layer(
                    batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TEST']
                )
            if self.training:
                if 'gt_boxes' in batch_dict: 
                    targets_dict = self.roi_head.assign_targets(batch_dict)
                    batch_dict['rois'] = targets_dict['rois']
                    batch_dict['roi_labels'] = targets_dict['roi_labels']
                    batch_dict['roi_targets_dict'] = targets_dict
                    num_rois_per_scene = targets_dict['rois'].shape[1]
                    if 'roi_valid_num' in batch_dict:
                        batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

            batch_dict = self.pfe(batch_dict)
            batch_dict = self.point_head(batch_dict)
            batch_dict = self.roi_head(batch_dict)

            if self.training:
                if 'gt_boxes' in batch_dict: 
                    loss, tb_dict, disp_dict = self.get_training_loss()

                    ret_dict = {
                        'loss': loss
                    }
                    return batch_dict, ret_dict, tb_dict, disp_dict
                else:
                    return batch_dict
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts
        
        else:
            raise Exception('Unsupprted model type')

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

class PVRCNNPlusPlus_M_DB(Detector3DTemplate_M_DB):
    def __init__(self, model_cfg, num_class, num_class_s2, dataset, dataset_s2, source_one_name):
        super().__init__(model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, dataset=dataset,
                        dataset_s2=dataset_s2, source_one_name=source_one_name)
        self.module_list = self.build_networks()
        self.source_one_name = source_one_name

    def forward(self, batch_dict):
        # Split the Concat dataset batch into batch_1 and batch_2
        split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, batch_dict)

        batch_s1 = {}
        batch_s2 = {}

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)

        # Dataset-specific DenseHead
        if len(split_tag_s1) == batch_dict['batch_size']:
            batch_dict = self.dense_head_s1(batch_dict)
            batch_dict = self.roi_head_s1.proposal_layer(
                batch_dict, nms_config=self.roi_head_s1.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )
            if self.training:
                targets_dict = self.roi_head_s1.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
                batch_dict['roi_targets_dict'] = targets_dict
                num_rois_per_scene = targets_dict['rois'].shape[1]
                if 'roi_valid_num' in batch_dict:
                    batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]
            batch_dict = self.pfe(batch_dict)
            batch_dict = self.point_head_s1(batch_dict)
            batch_dict = self.roi_head_s1(batch_dict)

        elif len(split_tag_s2) == batch_dict['batch_size']:
            batch_dict = self.dense_head_s2(batch_dict)
            batch_dict = self.roi_head_s2.proposal_layer(
                batch_dict, nms_config=self.roi_head_s2.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )
            if self.training:
                targets_dict = self.roi_head_s2.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
                batch_dict['roi_targets_dict'] = targets_dict
                num_rois_per_scene = targets_dict['rois'].shape[1]
                if 'roi_valid_num' in batch_dict:
                    batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]
            batch_dict = self.pfe(batch_dict)
            batch_dict = self.point_head_s2(batch_dict)
            batch_dict = self.roi_head_s2(batch_dict)

        else:
            batch_s1, batch_s2 = common_utils.split_two_batch_dict_gpu(split_tag_s1, split_tag_s2, batch_dict)
            # Split branch One:
            batch_s1 = self.dense_head_s1(batch_s1)
            batch_s1 = self.roi_head_s1.proposal_layer(
                batch_s1, nms_config=self.roi_head_s1.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )
            if self.training:
                targets_dict = self.roi_head_s1.assign_targets(batch_s1)
                batch_s1['rois'] = targets_dict['rois']
                batch_s1['roi_labels'] = targets_dict['roi_labels']
                batch_s1['roi_targets_dict'] = targets_dict
                num_rois_per_scene = targets_dict['rois'].shape[1]
                if 'roi_valid_num' in batch_s1:
                    batch_s1['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_s1['batch_size'])]
            batch_s1 = self.pfe(batch_s1)
            batch_s1 = self.point_head_s1(batch_s1)
            batch_s1 = self.roi_head_s1(batch_s1)

            # Split branch TWO:
            batch_s2 = self.dense_head_s2(batch_s2)
            batch_s2 = self.roi_head_s2.proposal_layer(
                batch_s2, nms_config=self.roi_head_s2.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )
            if self.training:
                targets_dict = self.roi_head_s2.assign_targets(batch_s2)
                batch_s2['rois'] = targets_dict['rois']
                batch_s2['roi_labels'] = targets_dict['roi_labels']
                batch_s2['roi_targets_dict'] = targets_dict
                num_rois_per_scene = targets_dict['rois'].shape[1]
                if 'roi_valid_num' in batch_s2:
                    batch_s2['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_s2['batch_size'])]
            batch_s2 = self.pfe(batch_s2)
            batch_s2 = self.point_head_s2(batch_s2)
            batch_s2 = self.roi_head_s2(batch_s2)            

        if self.training:
            split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, batch_dict)
            if len(split_tag_s1) == batch_dict['batch_size']:
                loss, tb_dict, disp_dict = self.get_training_loss_s1()
            
                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            elif len(split_tag_s2) == batch_dict['batch_size']:
                loss, tb_dict, disp_dict = self.get_training_loss_s2()
            
                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            else:
                loss_1, tb_dict_1, disp_dict_1 = self.get_training_loss_s1()
                loss_2, tb_dict_2, disp_dict_2 = self.get_training_loss_s2()
                ret_dict = {
                    'loss': loss_1 + loss_2
                }
                return ret_dict, tb_dict_1, disp_dict_1
        else:
            # NOTE: When peform the inference, only one dataset can be accessed.
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss_s1(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head_s1.get_loss()
        if self.point_head_s1 is not None:
            loss_point, tb_dict = self.point_head_s1.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head_s1.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
    
    def get_training_loss_s2(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head_s2.get_loss()
        if self.point_head_s2 is not None:
            loss_point, tb_dict = self.point_head_s2.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head_s2.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict