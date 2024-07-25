# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_15.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 15_attention_add
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_15.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 15supervised_adpt --supervised
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_30.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 30_attention_add
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_30.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 30supervised_adpt --supervised
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_45.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 45_attention_add
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_45.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 45supervised_adpt --supervised
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_60.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 60_attention_add
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_60.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 60supervised_adpt --supervised
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_75.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 75_attention_add
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_75.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 75supervised_adpt --supervised
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_90.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 90_attention_add
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_90.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 90supervised_adpt --supervised
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_105.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 105_attention_add
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_105.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 105supervised_adpt --supervised
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_120.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 120_attention_add
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_120.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 120supervised_adpt --supervised
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_135.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 135_attention_add
# bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_135.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --extra_tag 135supervised_adpt --supervised

# for num in 43 44 45 46 47 48 49 50 51; do
#     bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_150.yaml --ckpt ../output/waymo_models/second_res_150/150_attention_add_seed_$num/ckpt/checkpoint_epoch_30.pth
#     bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_150.yaml --ckpt ../output/waymo_models/second_res_150/150supervised_adpt_seed_$num/ckpt/checkpoint_epoch_30.pth
# done

# for scene in 135 120 105 90 75 60 45 30 15; do
    
#     for num in 43 44 45 46 47 48 49 50 51; do
#         bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}attention_add_seed_${num}"
#         bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --teacher_tag 32 --batch_size 12 --infos_seed $num --extra_tag "${scene}supervised_adpt_seed_${num}" --supervised
#     done
# done

for scene in 135 120; do
    
    for num in 43 44 45 46 47 48 49 50 51; do
        bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --ckpt ../output/waymo_models/second_res_$scene/${scene}attention_add_seed_$num/ckpt/checkpoint_epoch_30.pth
        bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_$scene.yaml --ckpt ../output/waymo_models/second_res_$scene/${scene}supervised_adpt_seed_$num/ckpt/checkpoint_epoch_30.pth
    done
done


# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_15.yaml --ckpt ../output/waymo_models/second_res_15/15_attention_add/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_15.yaml --ckpt ../output/waymo_models/second_res_15/15supervised_adpt/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_30.yaml --ckpt ../output/waymo_models/second_res_30/30_attention_add/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_30.yaml --ckpt ../output/waymo_models/second_res_30/30supervised_adpt/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_45.yaml --ckpt ../output/waymo_models/second_res_45/45_attention_add/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_45.yaml --ckpt ../output/waymo_models/second_res_45/45supervised_adpt/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_60.yaml --ckpt ../output/waymo_models/second_res_60/60_attention_add/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_60.yaml --ckpt ../output/waymo_models/second_res_60/60supervised_adpt/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_75.yaml --ckpt ../output/waymo_models/second_res_75/75_attention_add/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_75.yaml --ckpt ../output/waymo_models/second_res_75/75supervised_adpt/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_90.yaml --ckpt ../output/waymo_models/second_res_90/90_attention_add/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_90.yaml --ckpt ../output/waymo_models/second_res_90/90supervised_adpt/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_105.yaml --ckpt ../output/waymo_models/second_res_105/105_attention_add/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_105.yaml --ckpt ../output/waymo_models/second_res_105/105supervised_adpt/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_120.yaml --ckpt ../output/waymo_models/second_res_120/120_attention_add/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_120.yaml --ckpt ../output/waymo_models/second_res_120/120supervised_adpt/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_135.yaml --ckpt ../output/waymo_models/second_res_135/135_attention_add/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_135.yaml --ckpt ../output/waymo_models/second_res_135/135supervised_adpt/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_150.yaml --ckpt ../output/waymo_models/second_res_150/150_attention_add/ckpt/checkpoint_epoch_30.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/waymo_models/second_res_150.yaml --ckpt ../output/waymo_models/second_res_150/150supervised_adpt/ckpt/checkpoint_epoch_30.pth
