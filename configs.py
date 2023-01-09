import os
import os.path as osp

## General configurations
MASTER_PATH = '/mnt/shared_48tb/vbs'
EMBEDDING_PATH = osp.join(MASTER_PATH, 'embedding_features')

## Configurations for V3C dataset
V3C_KEYFRAME_DATA_PATH = "VBS2022"
V3C1_VIDEO_DATA_PATH = "V3C1_videos"
V3C2_VIDEO_DATA_PATH = "V3C2_videos"
V3C_IMAGE_NAME_PATH = osp.join(MASTER_PATH, f'V3C_filenames.txt')
V3C_DATASET_PATH = osp.join(V3C_KEYFRAME_DATA_PATH, 'keyframes')
V3C_FEATURE_DICT_PATH = osp.join(EMBEDDING_PATH, f'V3C_L14_336_features_128.pkl')

## Configurations for marine dataset
MARINE_PATH = "marine"
MARINE_VIDEO_PATH = "extracted/MarineVideoKit"
MARINE_IMAGE_NAME_PATH = osp.join(MASTER_PATH, f'marine_filenames.txt')
MARINE_DATASET_PATH = osp.join(MARINE_PATH, MARINE_VIDEO_PATH)
MARINE_FEATURE_DICT_PATH = osp.join(EMBEDDING_PATH, f'marine_L14_336_features_128.pkl')

## Configurations for Elasticsearch
""" Configuration for Elasticsearch """
ELASTIC_HOST = 'localhost'
ELASTIC_PORT = 9222
ELASTIC_INDEX = 'vbs2023'