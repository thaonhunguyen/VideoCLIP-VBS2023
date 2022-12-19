from dataset_helpers import *
from PIL import Image
import helpers

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="VBS22 Interactive Video Search Engine")
parser.add_argument('--query', '-q', type=str, default='wedding', help='Input query for searching')
parser.add_argument('--generate_features', '-g', default=True, help='Whether you want to generate features or not')
parser.add_argument('--feature_path', '-fp', help='Input the directory where you want to store the feature files')
parser.add_argument('--batch_size', '-b', type=int, default=16, help='Input batch size')
parser.add_argument('--dataset_name', '-d', type=str, default='V3C', help='Input dataset name')

def main(args):
    DATASET_NAME = args.dataset_name
    EMBEDDING_PATH = osp.join(DISK_PATH, VBS_PATH, 'embedding_features')
    FEATURE_DICT_PATH = osp.join(EMBEDDING_PATH, f'{DATASET_NAME}_L14_336_features_128.pkl')
    FEATURE_PATH = osp.join(EMBEDDING_PATH, f'{DATASET_NAME}_L14_336_features_128')
    
    print("Dataset name: ", DATASET_NAME)
    # clip = CLIPSearchEngine(src_path=DATASET_MASTER_PATH, feature_path=args.feature_path, batch_size=args.batch_size, generate_features=args.generate_features)
    if DATASET_NAME == 'marine':
        dataset_path = osp.join(MARINE_PATH, MARINE_VIDEO_PATH)
    elif DATASET_NAME == 'V3C':
        dataset_path = osp.join(V3C_KEYFRAME_DATA_PATH, 'keyframes')
    src_path = osp.join(DISK_PATH, VBS_PATH)
    image_name_path = osp.join(src_path, f'{DATASET_NAME}_filenames.txt')

    clip_model = CLIPSearchEngine(DATASET_NAME, src_path=osp.join(DISK_PATH, VBS_PATH), feature_path=FEATURE_DICT_PATH, generate_features=True, dataset_path=dataset_path, image_name_path=image_name_path)
    clip_model.encode_dataset(entire_dataset=True)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

# if __name__=='__main__':
#     data = dataset(dataset_name=DATASET_NAME)
#     if data.dataset_name == 'marine':
#         dataset_path = osp.join(DISK_PATH, VBS_PATH, MARINE_PATH, MARINE_VIDEO_PATH)
#     elif data.dataset_name == 'V3C':
#         dataset_path = osp.join(DISK_PATH, VBS_PATH, V3C_KEYFRAME_DATA_PATH, 'keyframes')
#     data.get_file_name(load_file=False, dataset_path=dataset_path)
#     print(data.image_names[:10])
#     clip_model = CLIPSearchEngine(data.dataset_name, src_path=osp.join(DISK_PATH, VBS_PATH), feature_path=FEATURE_PATH, generate_features=True)