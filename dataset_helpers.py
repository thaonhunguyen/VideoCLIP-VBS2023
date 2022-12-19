# from configs import *
# from helpers import *
from PIL import Image
from typing import List
from collections import defaultdict
from tqdm import tqdm

import torch
import pickle
import clip
import math
import joblib
import glob
import os
import os.path as osp
import numpy as np
import pandas as pd
import helpers


# DATASET_NAME = "V3C"
# DATASET_NAME = "marine"

MASTER_PATH = "/mnt/4TBSSD/ntnhu/"
DISK_PATH = "/mnt/shared_48tb/"
VBS_PATH = "vbs"
MARINE_PATH = "marine"
MARINE_VIDEO_PATH = "extracted/MarineVideoKit"
V3C_KEYFRAME_DATA_PATH = "VBS2022"
V3C1_VIDEO_DATA_PATH = "V3C1_videos"
V3C2_VIDEO_DATA_PATH = "V3C2_videos"



class dataset():
    def __init__(self, dataset_name='', src_path='', dataset_path=None, image_name_path=None):
        self.src_path = src_path
        self.image_names = None
        self.dataset_name = dataset_name
        self.image_name_path = image_name_path
        self.dataset_path = dataset_path
        if self.dataset_name == 'V3C':
            self.extension = '.png'
        elif self.dataset_name == 'marine':
            self.extension = '.jpg'

    def get_file_name(self, load_file=True):
        '''
        Function to get a list of images' names from the source path in ascending order
        
        params:
            - load_file: bool, default=True
                Whether load the available file of all image names or not
        '''
        if load_file==True:
            print("Loading all image names ...")
            print(self.image_name_path)
            # self.image_names = joblib.load(self.image_name_path)
            with open(self.image_name_path, 'r') as file:
                temp = file.read().splitlines()
            self.image_names = [osp.join(self.src_path, x) for x in temp]
            del temp
            
        else:
            print("Getting all image names from the source path ...")
            if self.dataset_name == 'marine':
                # filenames = glob.glob(osp.join(DATASET_MASTER_PATH, '*/*/*.jpg'), recursive=True)
                # with open(osp.join(DATASET_MASTER_PATH, 'excluded_images.txt'), 'rb') as file:
                #     excluded_images = file.read().splitlines()
                # image_names = [item for item in filenames if item not in excluded_images]
                # self.image_names = sort_list(image_names)
                # joblib.dump(self.image_names, IMAGE_NAME_PATH)
#                 self.image_names = filenames
                frame_path = osp.join(self.dataset_path, "information/selected_frames")
                frame_ids = helpers.sort_list(os.listdir(osp.join(self.src_path, frame_path)))
                self.image_names = [osp.join(frame_path, filename) for filename in frame_ids if self.extension in filename]
            elif self.dataset_name == 'V3C':
                self.image_names = []
                folder_names = helpers.sort_list(os.listdir(osp.join(self.src_path, self.dataset_path)))
                for folder in tqdm(folder_names):
                    folder_path = osp.join(self.dataset_path, folder)
                    filenames = helpers.sort_list(os.listdir(osp.join(self.src_path, folder_path)))
                    self.image_names.extend([osp.join(folder_path, filename) for filename in filenames if self.extension in filename])
            helpers.save_list_to_txt(self.image_names, self.image_name_path)
            self.image_names = [osp.join(self.src_path, x) for x in self.image_names]

# Define the CLIP encoding model class
class CLIP():
    def __init__(self, model_name='ViT-L/14@336px'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model, self.encoder = clip.load(self.model_name, device=self.device)

    def encode_images(self, stacked_images: torch.Tensor) -> np.ndarray:
        '''
        Function to transform images in the dataset into feature vectors
        
        params:
            - stacked_images: Tensor
                A stack of images to encode
        return:
            - List of feature vectors of the images
        ''' 
        with torch.no_grad():
            # Encode the images batch to compute the feature vectors and normalise them
            images_features = self.model.encode_image(stacked_images)
            images_features /= images_features.norm(dim=-1, keepdim=True)

        # Transfer the feature vectors back to the CPU and convert to numpy
        return images_features.cpu().numpy() 

    def encode_text_query(self, query:str) -> np.ndarray:
        '''
        Function to transform a text query string into vector
        
        params:
            - query: string
                An input string to encode
        
        return:
            - A numerical array of encoded text string
        '''
        with torch.no_grad():
            # Encode and normalise the description using CLIP
            text_encoded = self.model.encode_text(clip.tokenize(query).to(self.device))
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        
        return text_encoded


class CLIPSearchEngine():
    def __init__(self, dataset_name='', src_path='', feature_path='', dataset_path='', image_name_path='', batch_size=16, generate_features=False):
        self.dataset_name = dataset_name
        self.src_path = src_path
        self.dataset = dataset(dataset_name=self.dataset_name, src_path=self.src_path, dataset_path=dataset_path, image_name_path=image_name_path)
        self.clip_model = CLIP()
        self.feature_dict = {}
        self.features = None
        self.feature_path = feature_path
        self.batch_size = batch_size
        self.generate_features = generate_features

    def compute_clip_image_embeddings(self, image_batch: List[str]) -> defaultdict(list):
        '''
        Encoded image list into vectors using the pre-trained encoder loaded from CLIP
        
        params:
           - image_batch: List(str)
                A batch of images to encode
        
        return:
           - _: defaultdict
                Dictionary with keys are the image names and values are the embedding vectors
        '''
        # Sort the file name of all images in batch by
        image_batch = helpers.sort_list(image_batch)
        image_embeddings_dict = defaultdict(list)
        # Load all the images from the files          
        images = [Image.open(image_file) for image_file in image_batch]
        filenames = [helpers.convert_to_concepts(image_file, dataset_name=self.dataset_name)['filename'] for image_file in image_batch]
        # Encode all images
        images_encoded = torch.stack([self.clip_model.encoder(image) for image in images]).to(self.clip_model.device)
        image_embeddings = self.clip_model.encode_images(images_encoded)

        # Match file name with the embedding vectors
        for idx in range(len(image_batch)):
            image_embeddings_dict[filenames[idx]] = image_embeddings[idx]

        return image_embeddings_dict

    def encode_dataset(self, entire_dataset=True):
        '''
        Images will be divided into batches and encoded into embedding vectors.
        Then all embedding files will be saved for later use.
        
        params:
            - entire_dataset: bool, default=True
                Whether process the entire dataset or not
        returns:
            - _: defaultdict
                Dictionary with keys are the image names and values are the embedding vectors
        '''
        
        if self.dataset.image_names is None:
            print("Get all filenames")
            self.dataset.get_file_name()
        
        # Compute how many batches are needed
        if entire_dataset:
            print('Encode the whole dataset...')
            batches = math.ceil(len(self.dataset.image_names) / self.batch_size)
        else:
            print('Encode a subset of the dataset...')
            batches = 10
        
        if self.generate_features:
            self.feature_dict = {}
            print("Generate features ...")
            # Process each batch
            for i in tqdm(range(batches)):
            # for i in tqdm(range(10)):
#                 embedding_filename = osp.join(self.feature_path, f'{i:010d}.joblib')
# #                 try:
#                     # Select the images for the current batch
#                 batch_files = self.dataset.image_names[i*self.batch_size : (i+1)*self.batch_size]
#                 # Compute the features and save to a joblib file
#                 batch_embeddings = self.compute_clip_image_embeddings(batch_files)
#                 joblib.dump(batch_embeddings, embedding_filename)
#                 except:
#                     print(f"Problem with batch {i}.")
                batch_files = self.dataset.image_names[i*self.batch_size : (i+1)*self.batch_size]
                # Compute the features and save to a joblib file
                batch_embeddings = self.compute_clip_image_embeddings(batch_files)
                self.feature_dict.update(batch_embeddings)
            with open(osp.join(self.feature_path), 'wb') as file:
                pickle.dump(self.feature_dict, file)
        else:
            print("Load extracted features ...")
            self.load_features()

    @helpers.time_this
    def load_features(self):
        '''
        Load saved metadata files (encoded features)
        '''
#         try:
        print("Loading feature files ...")
# #             feature_list  = sort_list(glob(osp.join(self.feature_path, '*.joblib')))
#             feature_list = joblib.load(FEATURE_FILENAME_PATH)
#             for feature_file in tqdm(feature_list):
#                 feature = joblib.load(feature_file)
#                 self.feature_dict.update(feature)
#                 del feature
        with open(osp.join(self.feature_path), 'rb') as file:
            self.feature_dict = pickle.load(file)
        temp = self.feature_dict.values()
        self.features = np.asarray([*temp]).astype('float32')
        del temp
#         except:
#             print('There is no existing feature files.')
            
    def encode_input_query(self, query: str) -> np.array:
        '''
        Function to encode an input query into feature vector
        
        params:
            - query: str
                An input text query to search for the target images
                
        return:
            - _:array
                An embedded feature vector with shape (len, 1)
        '''
        if helpers.is_image(query):
            img_query = helpers.convert_to_concepts(query, dataset_name=self.dataset_name)['filename']
            feature = self.feature_dict[img_query]
            feature_vec = np.expand_dims(feature, axis=0)
            feature_vector = feature_vec.astype('float32')
        else:
            # Encode the string query into the latent space
            str_feature = self.clip_model.encode_text_query(query)
            feature_vector = str_feature.cpu().numpy().astype('float32')
        
        return feature_vector
        
    def search_query(self, query: str, num_matches=500, nlist=10, ss_type='faiss') -> List:
        '''
        Function to search for target images giving an input query
        
        params:
            - query: str
                An input text query to search for the target images
            - num_matches: integer, default=500
                The number of images matching the query
            - nlist: integer, default=10
                Index parameter for faiss (similarity search)
            - ss_type: str, default='faiss':
                Similarity search type, whether 'faiss' or 'cosine' similarity
        
        return:
            - A list of matching images to the input query
        '''
        if self.dataset.image_names is None:
            self.dataset.get_file_name()
            
        if self.features is None:
            self.load_features()

        # Encode the input query into feature vector
        feature_vector = self.encode_input_query(query)
        
        if ss_type == 'faiss':
            pass
            #dimension = self.features.shape[1]
            ## Initialise faiss searching object
            #quantiser = faiss.IndexFlatL2(dimension)  
            #index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_L2)
            #index.train(self.features) 
            #index.add(self.features)  
            ## Calculate the distances of the text vectors 
            #distances, indices = index.search(feature_vector, num_matches)
            #best_matched_image_names = [self.dataset.image_names[item] for item in indices[0]]
        else:
            # Compute the similarity between the description and each image using the Cosine similarity
            similarities = (feature_vector @ self.features.T).squeeze(0)
            indices = similarities.argsort()[-num_matches:][::-1]
            best_matched_image_names = [self.dataset.image_names[item] for item in indices]
            
        result = [helpers.convert_to_concepts(item, dataset_name=self.dataset_name) for item in best_matched_image_names]
        return result
    
def display_results(image_list=None, figsize=(15, 15), subplot_size=(5, 3)):
    '''
    Visualise images from the top most similar image list
    params:
        - image_list: List, default=None
            An input image list to display
        - figsize: tuple, default=(15, 15)
            The size of the figure to visualise
        - subplot_size: tuple, default=(5, 3)
            The size of the plot to visualise
    '''
    if image_list:
#         try:
        image_ids = [item['path'] for item in image_list]
        helpers.plot_figures(image_ids, figsize=figsize, subplot_size=subplot_size)
#         except:
#             print('Can\'t find best matched images.')


# if __name__=='__main__':
#     data = dataset(dataset_name=DATASET_NAME)
#     if data.dataset_name == 'marine':
#         dataset_path = osp.join(DISK_PATH, VBS_PATH, MARINE_PATH, MARINE_VIDEO_PATH)
#     elif data.dataset_name == 'V3C':
#         dataset_path = osp.join(DISK_PATH, VBS_PATH, V3C_KEYFRAME_DATA_PATH, 'keyframes')
#     data.get_file_name(load_file=False, dataset_path=dataset_path)
#     print(data.image_names[:10])
#     clip_model = CLIPSearchEngine(data.dataset_name, src_path=osp.join(DISK_PATH, VBS_PATH), feature_path=FEATURE_PATH, generate_features=True)