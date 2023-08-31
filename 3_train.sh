python 3_train.py \
    --data_root '/data/MaskedFaceRecog/data/20230608/train_crop' \
    --train_file '20230608_train_data_crop.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'config/backbone_conf.yaml' \
    --head_type 'MagFace' \
    --head_conf_file 'config/head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'train_result' \
    --epoches 18 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 10 \
    --momentum 0.9 \
    --log_dir 'logs' \
    --tensorboardx_logdir 'mv-hrnet' \
    2>&1 | tee train.log
