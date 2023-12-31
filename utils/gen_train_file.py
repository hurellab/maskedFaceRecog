"""
@author: Jun Wang
@date: 20210105
@contact: jun21wangustc@gmail.com
"""

import os

def gen_train_file(data_root, train_file):
    """Generate the train file, which has the following format.
    relative_path0 label0
    relative_path1 label1
    relative_path2 label2
    
    """
    train_file_buf = open(train_file, 'w')
    id_list = os.listdir(data_root)
    id_list.sort()
    for label, id_name in enumerate(id_list):
        cur_id_folder = os.path.join(data_root, id_name)
        cur_img_list = os.listdir(cur_id_folder)
        cur_img_list.sort()
        for index, image_name in enumerate(cur_img_list):
            cur_image_path = os.path.join(id_name, image_name)
            line = cur_image_path + ' ' + str(label)
            train_file_buf.write(line + '\n')

def gen_img_list_file(data_root, train_file):
    """Generate the train file, which has the following format.
    relative_path0
    relative_path1
    relative_path2
    
    """
    train_file_buf = open(train_file, 'w')
    id_list = os.listdir(data_root)
    id_list.sort()
    for id_name in id_list:
        cur_id_folder = os.path.join(data_root, id_name)
        cur_img_list = os.listdir(cur_id_folder)
        cur_img_list.sort()
        for index, image_name in enumerate(cur_img_list):
            cur_image_path = os.path.join(id_name, image_name)
            line = cur_image_path
            train_file_buf.write(line + '\n')
    
if __name__ == '__main__':
    data_root = '/data/FaceX-Zoo/training_mode/hurelData/data'
    # file to be generate.
    tain_file = 'hurelData_train_file.txt'
    gen_train_file(data_root, tain_file)
    img_list_file = 'hurelData_img_list.txt'
    gen_img_list_file(data_root, img_list_file)
