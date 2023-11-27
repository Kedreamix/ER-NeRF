# !/bin/bash

file=$1
if [ "$file" = "" ]; then
    file=main.py
fi
gpu=$2
if [ "$gpu" = "" ]; then
    gpu=0
fi
export CUDA_VISIBLE_DEVICES=$gpu

# 设置要运行的文件作为变量为 main.py

# 得到data文件夹下的文件夹
datasets=$(ls data | grep -v processed | grep -v processed2)
echo datasets: $datasets

test=true

# 遍历datasets
for dataset in $datasets
do
    # 如果是obama | Chinese | Eng1则跳过，设置一个列表
    if [ "$dataset" = "processed" ]; then
        continue
    fi
    echo train: $dataset
    loss="l1"
    if [ "$file" = "main.py" ]; then
        log_folder=logs/logs_triplane_error_$loss
        # python data_utils/process.py data/$dataset/$dataset.mp4
        # cp data/processed/$dataset.csv data/$dataset/au.csv
        # cp data/$dataset/aud.npy data/$dataset/aud_ds.npy
        mkdir -p $log_folder/trial_$dataset
        python $file data/$dataset --workspace $log_folder/trial_$dataset -O --iters 100000 --loss $loss --error_map
        python $file data/$dataset --workspace $log_folder/trial_$dataset -O --test --loss $loss --error_map
        cp -r $log_folder/trial_$dataset/checkpoints $log_folder/trial_$dataset/checkpoints_ --loss $loss --error_map
        python $file data/$dataset --workspace $log_folder/trial_$dataset -O --iters 125000 --finetune_lips --patch_size 32 --loss $loss --error_map
        python $file data/$dataset --workspace $log_folder/trial_$dataset -O --test --loss $loss --error_map
        # python main.py data/$dataset --workspace $log_folder/trial_$dataset"_torso" -O --torso --head_ckpt $log_folder/trial_$dataset/checkpoints/ngp.pth --iters 200000
        # python main.py data/$dataset --workspace l$log_folder/trial_$dataset"_torso" -O --torso --test
    fi
    
    if [ "$file" = "main_rad.py" ]; then
        log_folder=logs/logs_rad
        python $file data/$dataset --workspace $log_folder/trial_$dataset -O --iters 200000
        python $file data/$dataset --workspace $log_folder/trial_$dataset -O --test 
        python $file data/$dataset --workspace $log_folder/trial_$dataset -O --iters 250000 --finetune_lips --patch_size 32
        python $file data/$dataset --workspace $log_folder/trial_$dataset -O --test
        # python main.py data/$dataset --workspace $log_folder/trial_$dataset"_torso" -O --torso --head_ckpt $log_folder/trial_$dataset/checkpoints/ngp.pth --iters 200000
        # python main.py data/$dataset --workspace l$log_folder/trial_$dataset"_torso" -O --torso --test
    fi

    if [ "$file" = "main_tensorf.py" ]; then
        log_folder=logs/logs_tensorf
        python $file data/$dataset --workspace $log_folder/trial_$dataset -O --iters 100000
        python $file data/$dataset --workspace $log_folder/trial_$dataset -O --test
        cp -r $log_folder/trial_$dataset/checkpoints $log_folder/trial_$dataset/checkpoints_
        python $file data/$dataset --workspace $log_folder/trial_$dataset -O --iters 125000 --finetune_lips --patch_size 32
        python $file data/$dataset --workspace $log_folder/trial_$dataset -O --test
        # python main.py data/$dataset --workspace $log_folder/trial_$dataset"_torso" -O --torso --head_ckpt $log_folder/trial_$dataset/checkpoints/ngp.pth --iters 200000
        # python main.py data/$dataset --workspace l$log_folder/trial_$dataset"_torso" -O --torso --test
    fi
done
