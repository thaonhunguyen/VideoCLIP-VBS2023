import os
import sys 
sys.path.append(os.path.dirname(os.getcwd())) 

import torch
import json
import cv2
import helpers
# from utils import *
from dataset_helpers import *

from tqdm import tqdm
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="VBS22 Interactive Video Search Engine")
parser.add_argument('--dataset_name', '-d', type=str, default='V3C', help='Input dataset name')


def main(args):
    DATASET_NAME = args.dataset_name
    print("Dataset name: ", DATASET_NAME)
    # clip = CLIPSearchEngine(src_path=DATASET_MASTER_PATH, feature_path=args.feature_path, batch_size=args.batch_size, generate_features=args.generate_features)
    if DATASET_NAME == 'marine':
        dataset_path = osp.join(MARINE_PATH, MARINE_VIDEO_PATH)
    elif DATASET_NAME == 'V3C':
        dataset_path = osp.join(V3C_KEYFRAME_DATA_PATH, 'keyframes')
    src_path = osp.join(DISK_PATH, VBS_PATH)
    image_name_path = osp.join(src_path, f'{DATASET_NAME}_filenames.txt')
    
    data = dataset(dataset_name=DATASET_NAME, src_path=src_path, dataset_path=dataset_path, image_name_path=image_name_path)
    data.get_file_name()
    # print(len(data.image_names))

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_models/yolov5m.pt')

    for item in tqdm(data.image_names):
        concepts = helpers.convert_to_concepts(item, dataset_name=DATASET_NAME)
        # print(concepts)
        label_curr_path = osp.join(MASTER_PATH, 'VBS2023/VideoCLIP-VBS2023', 'object_detection/labels', DATASET_NAME, concepts['video'])
        image_curr_path = osp.join(MASTER_PATH, 'VBS2023/VideoCLIP-VBS2023', 'object_detection/images', DATASET_NAME, concepts['video'])
        detection = model(item) 
    #    temp = detection.save(image_curr_path)
        bbox = detection.pandas().xyxy[0]
        try:
            if not os.path.exists(label_curr_path):
                os.makedirs(label_curr_path)
            helpers.save_df_to_json(bbox, osp.join(label_curr_path, '{0}.json'.format(concepts['filename'])))
        except:
            print(item)
        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)