##FIX "head_conf.yaml -> num_class"
python 0_GenListFile.py \
    --print_train_list 1 \
    --print_test_list 1 \
    --train_data_root '/data/MaskedFaceRecog/data/20230517/train' \
    --train_data_list '20230517_train_data.txt' \
    --test_data_root '/data/MaskedFaceRecog/data/20230517/test' \
    --test_data_list '20230517_test_data.txt'