from dataset_helpers import *
from PIL import Image

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="VBS22 Interactive Video Search Engine")
parser.add_argument('--query', '-q', type=str, default='wedding', help='Input query for searching')
parser.add_argument('--generate_features', '-g', default=True, help='Whether you want to generate features or not')
parser.add_argument('--feature_path', '-fp', help='Input the directory where you want to store the feature files')
parser.add_argument('--batch_size', '-b', type=int, default=16, help='Input batch size')

def main(args):
    print("Dataset name: ", DATASET_NAME)
    # clip = CLIPSearchEngine(src_path=DATASET_MASTER_PATH, feature_path=args.feature_path, batch_size=args.batch_size, generate_features=args.generate_features)
    if DATASET_NAME == 'marine':
        dataset_path = osp.join(DISK_PATH, VBS_PATH, MARINE_PATH, MARINE_VIDEO_PATH)
    elif DATASET_NAME == 'V3C':
        dataset_path = osp.join(DISK_PATH, VBS_PATH, V3C_KEYFRAME_DATA_PATH, 'keyframes')
    image_name_path = osp.join(dataset_path, f'{DATASET_NAME}_filenames.txt')
    print(image_name_path)
    # with open(image_name_path, 'r') as file:
    #      temp = file.read().splitlines()
    # print(temp[:10])
    # data = dataset(dataset_name=DATASET_NAME, dataset_path=dataset_path, image_name_path=osp.join(image_filename_path))
    
    # print(image_filename_path)
    # data.get_file_name(load_file=True)
    # print(len(data.image_names))
    
    
    clip_model = CLIPSearchEngine(DATASET_NAME, src_path=osp.join(DISK_PATH, VBS_PATH), feature_path=FEATURE_DICT_PATH, generate_features=True, dataset_path=dataset_path, image_name_path=osp.join(image_name_path))
    print(clip_model.generate_features)
    clip_model.encode_dataset(entire_dataset=False)
    # print(len(clip_model.dataset.image_names))
    # print(len(clip_model.feature_dict))


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