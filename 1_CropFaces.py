import sys
sys.path.append('.')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import os
import yaml
import argparse
import cv2
import numpy as np
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from data_processor.train_dataset import ImageDataset

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

model_path = 'models'
scene = 'mask'

model_category = 'face_detection'
model_name =  model_conf[scene][model_category]
logger.info('Start to load the face detection model...')
# load model
try:
    faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
except Exception as e:
    logger.error('Failed to parse model configuration file!')
    logger.error(e)
    sys.exit(-1)
else:
    logger.info('Successfully parsed the model configuration file model_meta.json!')
try:
    model, cfg = faceDetModelLoader.load_model()
except Exception as e:
    logger.error('Model loading failed!')
    logger.error(e)
    sys.exit(-1)
else:
    logger.info('Successfully loaded the face detection model!')
faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)

model_category = 'face_alignment'
model_name =  model_conf[scene][model_category]
logger.info('Start to load the face landmark model...')
# load model
try:
    faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
except Exception as e:
    logger.error('Failed to parse model configuration file!')
    logger.error(e)
    sys.exit(-1)
else:
    logger.info('Successfully parsed the model configuration file model_meta.json!')
try:
    model, cfg = faceAlignModelLoader.load_model()
except Exception as e:
    logger.error('Model loading failed!')
    logger.error(e)
    sys.exit(-1)
else:
    logger.info('Successfully loaded the face landmark model!')
faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='crop the faces from the data')
    conf.add_argument("--img_data_root", type = str, 
                  help = "path to the root of img data")
    conf.add_argument("--img_list", type = str, 
                  help = "file name of the img list")
    conf.add_argument("--output_dir", type = str, 
                  help = "path to save cropped images")
    conf.add_argument("--threshold", type = float, default=0.9,
                  help = "threshold for FaceDetModelHandler")
    args = conf.parse_args()   
    threshold = args.threshold
    face_cropper = FaceRecImageCropper()

    # read image
    img_file_buf = open(args.img_list)
    tmp_img_name = "temp/temp.jpg"
    line = img_file_buf.readline().strip()
    id=0
    skip=0
    while line:
        image_path, image_label = line.split(' ')
        read_image_path = os.path.join(args.img_data_root, image_path)
        image = cv2.imread(read_image_path, cv2.IMREAD_COLOR)
        aspect_ratio = image.shape[1] / image.shape[0]
        image = cv2.resize(image, (int(700*aspect_ratio), 700))

        try:
            dets = faceDetModelHandler.inference_on_image(image)
        except Exception as e:
           logger.error('Face detection failed!')
           logger.error(e)
           sys.exit(-1)
        
        if dets.size > 0:
            try:
                image_save = image.copy()
                for det in dets:
                    if det[4]<threshold: continue
                    landmarks = faceAlignModelHandler.inference_on_image(image, det)
                    for (x, y) in landmarks.astype(np.int32):
                        cv2.circle(image_save, (x, y), 2, (255, 0, 0),-1)
                    cv2.imwrite(tmp_img_name, image_save)
                    landmarks = landmarks.ravel().astype(np.int32)
                    cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
                    save_path_img = os.path.join(args.output_dir, image_label, str(id)+".jpg")
                    id += 1
                    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)
                    cv2.imwrite(save_path_img, cropped_image)
                    logger.info(f"Image saved at: {save_path_img}")
            except Exception as e:
                logger.error('Face landmark failed!')
                logger.error(e)
                sys.exit(-1)
        else:
            skip += 1

        line = img_file_buf.readline().strip()
    print( skip, " images are skipped!")
    print("Crop is done. Please erase noises and generate train data for cropped images")
