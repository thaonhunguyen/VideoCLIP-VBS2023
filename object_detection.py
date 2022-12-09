import os
import sys 
sys.path.append(os.path.dirname(os.getcwd())) 

from helpers import *
from dataset_helpers import *
import torch
import json
import cv2

from tqdm import tqdm
from PIL import Image

data = dataset(dataset_name='V3C')
data.get_file_name()

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_models/yolov5m.pt')

for item in tqdm(data.image_names):
    concepts = convert_to_concepts(item)
    label_curr_path = osp.join(DATASET_MASTER_PATH, 'object_detection/labels', concepts['video'])
    image_curr_path = osp.join(DATASET_MASTER_PATH, 'object_detection/images', concepts['video'])
    detection = model(item) 
#    temp = detection.save(image_curr_path)
    bbox = detection.pandas().xyxy[0]
    try:
        save_df_to_json(bbox, osp.join(label_curr_path, '{0}.json'.format(concepts['filename'][:-4])))
    except:
        print(item)