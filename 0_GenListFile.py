import os
import argparse

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
        print("\t", id_name, label)
        cur_id_folder = os.path.join(data_root, id_name)
        cur_img_list = os.listdir(cur_id_folder)
        cur_img_list.sort()
        for index, image_name in enumerate(cur_img_list):
            if not image_name.endswith(('.png', '.jpg', '.jpeg')): continue
            cur_image_path = os.path.join(id_name, image_name)
            line = cur_image_path + ' ' + str(label)
            train_file_buf.write(line + '\n')

def gen_list_file(data_root, list_file):
    """Generate the train file, which has the following format.
    relative_path0 label0
    relative_path1 label1
    relative_path2 label2
    
    """
    train_file_buf = open(list_file, 'w')
    id_list = os.listdir(data_root)
    id_list.sort()
    for id_name in id_list:
        print("\t", id_name)
        cur_id_folder = os.path.join(data_root, id_name)
        cur_img_list = os.listdir(cur_id_folder)
        cur_img_list.sort()
        for index, image_name in enumerate(cur_img_list):
            if not image_name.endswith(('.png', '.jpg', '.jpeg')): continue
            cur_image_path = os.path.join(id_name, image_name)
            train_file_buf.write(cur_image_path + '\n')
    
if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='prepare file lists')
    conf.add_argument("--print_train_list", type = int, 
                      help = "To print train list file")
    conf.add_argument("--print_test_list", type = int, 
                      help = "To print test list file")
    conf.add_argument("--train_data_root", type = str, 
                      help = "Path to the train data")
    conf.add_argument("--train_data_list", type = str, 
                      help = "File name of the train_data_list")
    conf.add_argument("--test_data_root", type = str, 
                      help = "Path to the test data")
    conf.add_argument("--test_data_list", type = str, 
                      help = "File name of the test_data_list")
    args = conf.parse_args()

    # file to be generate.
    if args.print_train_list>0:
        print("train file ID:")
        gen_train_file(args.train_data_root, args.train_data_list)
    if args.print_test_list>0:
        print("test file ID:")
        gen_list_file(args.test_data_root, args.test_data_list)
