import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from tqdm import tqdm 


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    # ours 
    parser.add_argument('--cfg_file', type=str, default='cfgs/waymo_models/second_res_90.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default="/root/Desktop/workspace/3DTrans/output/waymo_models/second_res_60/60attention_add_new_gt_new_sample_46/ckpt/checkpoint_epoch_30.pth")

    # default (cbgs)
    parser.add_argument('--default-ckpt', type=str, default="/root/Desktop/workspace/3DTrans/output/waymo_models/second_res_60/60supervised_adpt_seed_46/ckpt/checkpoint_epoch_30.pth", help='specify the pretrained model')

    # parser.add_argument('--default-cbgs-ckpt', type=str, default="/root/Desktop/workspace/from_kauai_data1_goodgpt/goodgpt/ObjDet/daeun/OpenPCDet_kitti/output/cfgs/nuscenes_models/cbgs_second_multihead_nds6229_updated.pth", 
    #                     help='specify the pretrained model')


    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=None, workers=4,
        logger=logger,
        training=False,
        merge_all_iters_to_one_epoch=False,
        total_epochs=None
    )

    logger.info(f'Total number of samples: \t{len(train_set)}')

    ## our model 
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    ## default model 
    d_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    d_model.load_params_from_file(filename=args.default_ckpt, logger=logger, to_cpu=True)
    d_model.cuda()
    d_model.eval()

    ## cbgs model 
    # c_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    # c_model.load_params_from_file(filename=args.default_cbgs_ckpt, logger=logger, to_cpu=True)
    # c_model.cuda()
    # c_model.eval()
    print('model evaluation start!')

    # consider_list = [750, 700, 320, 200]
    consider_list = [3323, 3423, 3524]
    #400, 413 
    with torch.no_grad():

        for idx, data_dict in tqdm(enumerate(train_set)):

            # if idx % 10 == 0 and idx > 1170:
            # if idx == 750 :
            if idx in consider_list : 
                print(idx)
                logger.info(f'Visualized sample index: \t{idx + 1}')

                # data prep 
                data_dict = train_set.collate_batch([data_dict])
                # print(data_dict['gt_boxes'].shape)
                # print('=' * 50)

                load_data_to_gpu(data_dict)
                

                """
                1. Our viz
                """
                print("Our_viz start! ")
                pred_dicts, _ = model.forward(data_dict)
                # print(pred_dicts[0]['pred_boxes'].shape)
                
                
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], 
                    gt_boxes = data_dict['gt_boxes'].reshape(data_dict['gt_boxes'].shape[1], -1)[:, :9],    # added by me.. 
                    ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], 
                    ref_labels=pred_dicts[0]['pred_labels'],
                )
                #import pdb; pdb.set_trace()
                # PyTorch 텐서를 NumPy 배열로 변환
                #dict1 = {'points': data_dict['points'][:, 1:].cpu().numpy(), 'gt_boxes':data_dict['gt_boxes'].reshape(data_dict['gt_boxes'].shape[1], -1)[:, :9].cpu().numpy(), 'pred_boxes':pred_dicts[0]['pred_boxes'].cpu().numpy(), 'pred_labels':pred_dicts[0]['pred_labels'].cpu().numpy()}
                
                #np.save('data_dict_400_ours.npy', dict1)
                
                if not OPEN3D_FLAG:
                    mlab.savefig(filename= './results/' + str(idx) + '_test_our.png')
                    print('saved Our figure')
                    # mlab.show(stop=True)


                """
                2. Default Pointpillar
                """
                print("Default start! ")
                pred_dicts, _ = d_model.forward(data_dict)
                print(pred_dicts[0]['pred_labels'])
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], 
                    gt_boxes = data_dict['gt_boxes'].reshape(data_dict['gt_boxes'].shape[1], -1)[:, :9],    # added by me.. 
                    ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], 
                    ref_labels=pred_dicts[0]['pred_labels'],     # pred한 label
                )
                #dict2 = {'points': data_dict['points'][:, 1:].cpu().numpy(), 'gt_boxes':data_dict['gt_boxes'].reshape(data_dict['gt_boxes'].shape[1], -1)[:, :9].cpu().numpy(), 'pred_boxes':pred_dicts[0]['pred_boxes'].cpu().numpy(), 'pred_labels':pred_dicts[0]['pred_labels'].cpu().numpy()}

                #np.save('data_dict_400_baseline.npy', dict2)
                if not OPEN3D_FLAG:
                    mlab.savefig(filename= './results/' + str(idx) + '_test_default.png')
                    print('saved default figure')
                    # mlab.show(stop=True)

                
                """
                3. CBGS
                """
                # print("CBGS start! ")
                # pred_dicts, _ = c_model.forward(data_dict)
                # print(pred_dicts[0]['pred_labels'])

                # V.draw_scenes(
                #     points=data_dict['points'][:, 1:], 
                #     gt_boxes = data_dict['gt_boxes'].reshape(data_dict['gt_boxes'].shape[1], -1)[:, :9],    # added by me.. 
                #     ref_boxes=pred_dicts[0]['pred_boxes'],
                #     ref_scores=pred_dicts[0]['pred_scores'], 
                #     ref_labels=pred_dicts[0]['pred_labels'],     # pred한 label
                # )

                # if not OPEN3D_FLAG:
                #     mlab.savefig(filename= './results/' + str(idx) + '_test_cbgs.png')
                #     print('saved cbgs figure')
                #     # mlab.show(stop=True)

            else:
                continue
            # break

    logger.info('Demo done.')


if __name__ == '__main__':
    main()