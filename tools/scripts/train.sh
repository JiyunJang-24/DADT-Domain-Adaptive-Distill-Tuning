
# for scene in 30 60 90 120; do
#     for num in 43 44 45 46 47; do
#         bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}attention_add_new_gt_new_sample_${num}"
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --ckpt ../output/waymo_models/pv_rcnn_plusplus_resnet_$scene/${scene}attention_add_new_gt_new_sample_$num/ckpt/checkpoint_epoch_30.pth --batch_size 32
#         bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}supervised_adpt_seed_${num}" --supervised
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --ckpt ../output/waymo_models/pv_rcnn_plusplus_resnet_$scene/${scene}supervised_adpt_seed_$num/ckpt/checkpoint_epoch_30.pth --batch_size 32
#     done
# done

# for num in 43 44 45; do
#     for scene in 60 30; do
#         #bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}new_gt_rereprod_new_sample_${num}"
#         #bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --ckpt ../output/waymo_models/second_res_$scene/${scene}new_gt_rereprod_new_sample_$num/ckpt/checkpoint_epoch_30.pth
#         #bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}supervised_adpt_seed_${num}" --supervised
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --ckpt ../output/waymo_models/pv_rcnn_plusplus_resnet_$scene/${scene}supervised_adpt_seed_$num/ckpt/checkpoint_epoch_30.pth
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --ckpt ../output/waymo_models/pv_rcnn_plusplus_resnet_$scene/${scene}attention_add_new_gt_new_sample_$num/ckpt/checkpoint_epoch_30.pth

#     done
# done

for num in 46 47 48; do
    for scene in 60 30; do
        bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}attention_add_new_gt_new_sample_${num}"
        bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}supervised_adpt_seed_${num}" --supervised
        bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --ckpt ../output/waymo_models/pv_rcnn_plusplus_resnet_$scene/${scene}attention_add_new_gt_new_sample_$num/ckpt/checkpoint_epoch_30.pth
        bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --ckpt ../output/waymo_models/pv_rcnn_plusplus_resnet_$scene/${scene}supervised_adpt_seed_$num/ckpt/checkpoint_epoch_30.pth

    done
done

#bash scripts/dist_train.sh 4 --

#bash scripts/UDA/dist_train_uda.sh 4 --cfg_file cfgs/waymo_models/second_res_semi_st3d_3%_adam.yaml --pretrained_model ../output/waymo_models/second_res_120/120supervised_adpt_seed_43/ckpt/checkpoint_epoch_30.pth --batch_size 24 --fix_random_seed --extra_tag "3%semi_adpt_lr1000_ep10_st3d_adam_all_neg25_ps_5" 
# for num in 5 4 3 2 1; do
#     bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_90.yaml --ckpt ../output/waymo_models/second_res_semi_st3d_3%_adam/3%semi_adpt_lr1000_ep10_st3d_adam_all_neg25_ps_5/ckpt/checkpoint_epoch_$num.pth
# done



#bash scripts/UDA/dist_train_uda.sh 4 --cfg_file cfgs/waymo_models/second_res_semi_st3d_10000_adam.yaml --pretrained_model ../output/waymo_models/second_res_120/120supervised_adpt_seed_43/ckpt/checkpoint_epoch_30.pth --batch_size 24 --fix_random_seed --extra_tag "10000semi_adpt_lr1000_ep10_st3d_adam_all_neg25_ps_5" 
#ash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_90.yaml --ckpt ../output/waymo_models/second_res_semi_st3d_10000_adam/10000semi_attention_lr1000_ep20_st3d_adam_all_neg25_ps_4/ckpt/checkpoint_epoch_$num.pth

# for num in 1 2 3 4; do
#     bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_90.yaml --ckpt ../output/waymo_models/second_res_semi_st3d_10000/10000semi_attention_lr1000_ep10_st3d_all_neg25_ps_5/ckpt/checkpoint_epoch_$num.pth
#     #bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_90.yaml --ckpt ../output/waymo_models/second_res_semi_st3d_10000/10000semi_adpt_lr100_ep10_st3d_all_neg25_ps_2/ckpt/checkpoint_epoch_$num.pth
# done

# for scene in 90; do
#     for num in 43 44 45; do
#         #bash scripts/dist_train.sh 4 --cfg_file cfgs/DA/waymo_kitti/source_only/pv_rcnn_plus_sn_kitti_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --batch_size 12 --infos_seed $num --extra_tag "${scene}kitti_DA_sample_${num}"
#         #bash scripts/dist_train.sh 4 --cfg_file cfgs/DA/waymo_kitti/source_only/pv_rcnn_plus_sn_kitti_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --batch_size 12 --infos_seed $num --extra_tag "${scene}attention_add_new_gt_kitti_DA_sample_${num}"

#         #bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --ckpt ../output/waymo_models/second_res_$scene/${scene}attention_add_new_gt_beam64_new_sample_$num/ckpt/checkpoint_epoch_30.pth --batch_size 32
#         #bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/DA/waymo_kitti/source_only/pv_rcnn_plus_sn_kitti_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}attention_add_new_gt_kitti_DA_sample_${num}"
#         #bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --ckpt ../output/waymo_models/second_res_$scene/${scene}attention_add_new_gt_beam64_new_sample_$num/ckpt/checkpoint_epoch_30.pth --batch_size 32
#         #bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}supervised_adpt_seed_${num}" --supervised
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --ckpt ../output/waymo_models/pv_rcnn_plusplus_resnet_$scene/${scene}attention_add_new_gt_new_sample_$num/ckpt/checkpoint_epoch_30.pth
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --ckpt ../output/waymo_models/pv_rcnn_plusplus_resnet_$scene/${scene}supervised_adpt_seed_$num/ckpt/checkpoint_epoch_30.pth
#     done
# done



# for num in 43 44 45; do
#     for scene in 1 2 3 4 5; do
#         bash scripts/dist_mimic_nuS_train.sh 4 --cfg_file cfgs/nuscenes_models/cbgs_second_multihead_${scene}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 32 --infos_seed $num --extra_tag "${scene}%attention_add_new_gt_sample_${num}" 
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/nuscenes_models/cbgs_second_multihead_${scene}%.yaml --ckpt ../output/nuscenes_models/cbgs_second_multihead_${scene}%/${scene}%attention_add_new_gt_sample_$num/ckpt/checkpoint_epoch_20.pth --batch_size 128

#         bash scripts/dist_mimic_nuS_train.sh 4 --cfg_file cfgs/nuscenes_models/cbgs_second_multihead_${scene}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 32 --infos_seed $num --extra_tag "${scene}%supervised_adpt_sample_${num}" --supervised
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/nuscenes_models/cbgs_second_multihead_${scene}%.yaml --ckpt ../output/nuscenes_models/cbgs_second_multihead_${scene}%/${scene}%supervised_adpt_sample_$num/ckpt/checkpoint_epoch_20.pth --batch_size 128
#         #bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/DA/waymo_kitti/source_only/pv_rcnn_plus_sn_kitti_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --EMA 0.9997 --extra_tag "${scene}attention_add_new_gt_kitti_DA_EMA_9997_sample_${num}"
#         #bash scripts/dist_train.sh 4 --cfg_file cfgs/DA/waymo_kitti/source_only/pv_rcnn_plus_sn_kitti_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --batch_size 12 --infos_seed $num --extra_tag "${scene}attention_add_new_gt_kitti_DA_EMA_9997_sample_${num}"
        
#     done
# done

# bash scripts/dist_mimic_train_semi.sh 4 --cfg_file cfgs/waymo_models/second_res_semi_1000.yaml --pretrained_model ../output/waymo_models/second_res_120/120attention_add_new_gt_new_sample_43/ckpt/checkpoint_epoch_30.pth --pretrained_teacher_model ../output/waymo_models/second_res_120/120attention_add_new_gt_new_sample_43/ckpt/checkpoint_epoch_30.pth --batch_size 24 --fix_random_seed --EMA 0.9997 --supervised --extra_tag "1000semi_120attention_sample_43_lr_10_confid03_EMA9997epoch" 
# for num in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
#     bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_semi_1000.yaml --ckpt ../output/waymo_models/second_res_semi_1000/1000semi_120attention_sample_43_lr_10_confid03_EMA9997epoch/ckpt/checkpoint_epoch_${num}.pth
# done
#bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_semi_1000.yaml --ckpt ../output/waymo_models/second_res_semi_1000/1000semi_90attention_sample_44_epoch/ckpt/checkpoint_epoch_1.pth



# for num in 43; do
#     for scene in 30; do
#         #bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}attention_add_new_gt_new_sample_${num}"
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --ckpt ../output/waymo_models/pv_rcnn_plusplus_resnet_$scene/${scene}attention_add_new_gt_new_sample_$num/ckpt/checkpoint_epoch_30.pth --batch_size 32
#         #bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}supervised_adpt_seed_${num}" --supervised
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_resnet_$scene.yaml --ckpt ../output/waymo_models/pv_rcnn_plusplus_resnet_$scene/${scene}supervised_adpt_seed_$num/ckpt/checkpoint_epoch_30.pth --batch_size 32
#     done
# done





# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_1%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 24 --extra_tag "1%_attention_add_self_norm"
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_1%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 24 --extra_tag "1%_supervised_adpt" --supervised
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_1%.yaml --ckpt ../output/waymo_models/second_res_1%/1%_attention_add_self_norm/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_1%.yaml --ckpt ../output/waymo_models/second_res_1%/1%_supervised_adpt/ckpt/checkpoint_epoch_30.pth



# for scene in 60 30; do
    
#     for num in 43 44 45 46 47; do
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --ckpt ../output/waymo_models/second_res_$scene/${scene}attention_add_seed_$num/ckpt/checkpoint_epoch_30.pth
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --ckpt ../output/waymo_models/second_res_$scene/${scene}supervised_adpt_seed_$num/ckpt/checkpoint_epoch_30.pth
#     done
# done




# for scene in 90; do
#     for num in 43 44 45; do
#         bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --EMA 0.9997 --extra_tag "${scene}attention_add_new_gt_EMA_9997_sample_${num}"
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --ckpt ../output/waymo_models/second_res_$scene/${scene}attention_add_new_gt_EMA_9997_sample_$num/ckpt/checkpoint_epoch_30.pth

#         #bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}supervised_adpt_seed_${num}" --supervised
#     done
# done

# for scene in 75 60 45 30 15; do
#     for num in 43 44 45 46 47; do
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --ckpt ../output/waymo_models/second_res_$scene/${scene}attention_add_seed_$num/ckpt/checkpoint_epoch_30.pth
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --ckpt ../output/waymo_models/second_res_$scene/${scene}supervised_adpt_seed_$num/ckpt/checkpoint_epoch_30.pth
#     done
# done

