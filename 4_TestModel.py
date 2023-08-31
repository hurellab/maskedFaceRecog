import numpy as np
import torch
import torch.nn.functional as F
import argparse
import sys
import os
import cv2
import yaml
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from data_processor.test_dataset import CommonTestDataset
from torch.utils.data import Dataset, DataLoader
from utils.extractor.feature_extractor import CommonExtractor
from backbone.backbone_def import BackboneFactory
from utils.model_loader import ModelLoader

def TransformImg(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, encoded_image = cv2.imencode('.png', image)
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_UNCHANGED)
    decoded_image = (decoded_image.transpose((2, 0, 1)) - 127.5) / 128.
    decoded_image = torch.from_numpy(decoded_image.astype(np.float32))
    return decoded_image

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
    conf.add_argument("--model_file", type = str, 
                  help = "model file")
    conf.add_argument("--test_data_root", type = str, 
                  help = "path to the test data")
    conf.add_argument("--test_data_file", type = str, 
                  help = "list file for the test data")
    conf.add_argument("--result_dir", type = str, 
                  help = "path to save result image")
    conf.add_argument("--threshold", type = float, default=0.9, 
                  help = "threshold for FaceDetModelHandler")
    args = conf.parse_args()
    threshold = args.threshold
    face_cropper = FaceRecImageCropper()

    text = 'Sample Text'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    text_color = (0, 255, 0)
    text_thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, text_thickness)

    model = torch.load(args.model_file)
    logger.info('loaded trained model')
    prototype = model['state_dict']['head.weight']
    prototype = torch.transpose(prototype.to('cuda:0'), 0, 1)
    logger.info('configured prototype')

    data_loader = DataLoader(CommonTestDataset(args.test_data_root, args.test_data_file, False), 
                        batch_size=10240, num_workers=4, shuffle=False)
    backbone_factory = BackboneFactory('MobileFaceNet', 'config/backbone_conf.yaml')
    model_loader = ModelLoader(backbone_factory)
    model = model_loader.load_model(args.model_file)
    feature_extractor = CommonExtractor('cuda:0')
    image_name2feature = feature_extractor.extract_online(model, data_loader)

    for idx in image_name2feature:
        features = torch.tensor(image_name2feature[idx]).to('cuda:0')
        similarities = F.cosine_similarity(features.unsqueeze(0), prototype, dim=1)
        predicted_class = torch.argmax(similarities)
        print(idx, predicted_class)

    # read image
    # img_file_buf = open(args.test_data_file)
    # line = img_file_buf.readline().strip()
    # skip = 0
    # while line:
    #     image_path, image_label = line.split(' ')
    #     read_image_path = os.path.join(args.test_data_root, image_path)
        
    #     # image = cv2.imread(read_image_path, cv2.IMREAD_COLOR)
    #     # aspect_ratio = image.shape[1] / image.shape[0]
    #     # image = cv2.resize(image, (int(700*aspect_ratio), 700))

    #     try:
    #         dets = faceDetModelHandler.inference_on_image(image)
    #     except Exception as e:
    #        logger.error('Face detection failed!')
    #        logger.error(e)
    #        sys.exit(-1)

    #     images = []    
    #     if dets.size > 0:
    #         try:
    #             image_save = image.copy()
    #             for det in dets:
    #                 if det[4]<threshold: continue
    #                 landmarks = faceAlignModelHandler.inference_on_image(image, det) \
    #                             .ravel().astype(np.int32)
    #                 cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
    #                 cropped_image = TransformImg(cropped_image)
    #                 images.append(cropped_image)
    #                 print(det)
    #                 # cv2.rectangle(image_save, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 2)
    #             images = torch.cat(images, dim=2)
    #             images = images.to('cuda:0')
    #             features = model(images)
    #             print(1)
    #             features = F.normalize(features)
    #             print(2)
    #             features = features.cpu().numpy()
    #             print(3)
    #             similarities = F.cosine_similarity(features.unsqueeze(0), prototype, dim=1)
    #             print(similarities)
    #             predicted_class = torch.argmax(similarities, dim=1)
    #             print(image_label, predicted_class)
    #         except Exception as e:
    #             logger.error('Face landmark failed!')
    #             logger.error(e)
    #             sys.exit(-1)
    #     else:
    #         skip += 1
    # print( skip, " images are skipped!")    

