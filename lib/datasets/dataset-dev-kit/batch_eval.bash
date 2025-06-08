#!/bin/bash

# model_list=("monodetr_ema_yolov9_match_2D_" )
# # model_list=("monodetr_ema_yolov9_e2" "monodetr_ema_v9_" "monodetr_ema_yolov9_" "monodetr_ema_yolov9_e")
# # threshold_list=(2.0)
# threshold_list=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0)
# # threshold_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)
# camera_list=("south1" "south2")
# scene_list=("all")
# # scene_list=("day" "night" "all")

# for model in "${model_list[@]}"; do
#     echo "Model: $model"
#     output_base="/workspace/MonoDETR/outputs/${model}"
#     mkdir -p "evaluation_result_${model}"
#     for threshold in "${threshold_list[@]}"; do
#         echo "Threshold: $threshold"
#         for camera_id in "${camera_list[@]}"; do
#             for scene in "${scene_list[@]}"; do
#                 output_dir="${output_base}${threshold}/outputs/${camera_id}"
#                 echo "Running evaluation for Camera ID: $camera_id, Scene: $scene with Output Dir: $output_dir"
#                 CUDA_VISIBLE_DEVICES=2 python src/eval/evaluation.py --camera_id "$camera_id" --output_dir "$output_dir" --scene "$scene" > "evaluation_result_${model}/evaluation_${threshold}_${camera_id}_${scene}.log"
#             done
#         done
#     done
# done

model_list=("monodetr_bl_tum_feature_extrinsic_uf")
camera_list=("south1" "south2")
scene_list=("all")

for model in "${model_list[@]}"; do
    echo "Model: $model"
    output_base="/workspace/MonoDETR/outputs/${model}"
    mkdir -p "evaluation_result_${model}"
    for camera_id in "${camera_list[@]}"; do
        for scene in "${scene_list[@]}"; do
            output_dir="${output_base}${threshold}/outputs/${camera_id}"
            echo "Running evaluation for Camera ID: $camera_id, Scene: $scene with Output Dir: $output_dir"
            CUDA_VISIBLE_DEVICES=2 python src/eval/evaluation.py --camera_id "$camera_id" --output_dir "$output_dir" --scene "$scene" > "evaluation_result_${model}/evaluation_${camera_id}_${scene}.log"
        done
    done
done