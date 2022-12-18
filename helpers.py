import re 
import json
import os
import os.path as osp
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
from tqdm import tqdm
# from bs4 import BeautifulSoup
from itertools import islice


### -------------------------------------------------------------------------- ###
###                              SORT FUNCTIONS                                ###
### -------------------------------------------------------------------------- ###

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', str(text)) ]

def sort_list(input_list):
    '''
    Function to convert an input list into a list in ascending order based on natual keys

    params:
        - input_list: List
    '''
    input_list.sort(key=natural_keys)
    return input_list

def sort_dict(input_dict, by_value=True, descending=False):
    index = 1
    if not by_value:
        index = 0
    return dict(sorted(input_dict.items(), key=lambda item: item[index], reverse=descending))

def merge_two_dicts(first_dict, second_dict):
    merged_dict = first_dict.copy()   
    merged_dict.update(second_dict)   
    return merged_dict


def get_key_by_value(input_dict, value):
    '''
    Get a list of keys from dictionary which has the given value
    '''
    key_list = list()
    item_lit = input_dict.items()
    for item  in item_lit:
        if item[1] == value:
            key_list.append(item[0])
    return key_list


### -------------------------------------------------------------------------- ###
###                                FILE I/O                                    ###
### -------------------------------------------------------------------------- ###

def save_df_to_json(data_df: pd.DataFrame, filename, orient='records', indent=4):
    data = data_df.to_json(orient=orient) 
    parsed_data = json.loads(data)    
    with open(filename, 'w') as f:
        json.dump(parsed_data, f, indent=indent)

def save_dict_to_json(data: dict, filename, orient='records', indent=4):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=indent)

def save_list_to_csv(obj: List, filename, delimiter='\n'):
    with open(filename, 'w') as f:
        # create the csv writer
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(obj)
        
def save_list_to_txt(obj: List, filename):
    with open(filename, 'w') as f:
        for item in obj:
            # write each item on a new line
            f.write("%s\n" %item)

def load_json(json_file, is_list=False):
    with open(json_file, 'r') as file:
        text_data = file.read()
        text_data = '[' + re.sub(r'\}\s\{', '},{', text_data) + ']'
        json_data = json.loads(text_data)
        if is_list:
            return json_data
    return json_data[0]


# def load_xml(xml_file):
#     with open(xml_file, 'r') as file:
#         contents = file.read()
#     soup = BeautifulSoup(contents, 'xml')
#     return soup


### -------------------------------------------------------------------------- ###
###                             HELPER FUNCTION                                ###
### -------------------------------------------------------------------------- ###

def time_this(func):
    def calc_time(*args, **kwargs):
        before = datetime.datetime.now()
        x = func(*args, **kwargs)
        after = datetime.datetime.now()
        print("Function {} elapsed time: {}".format(func.__name__, after-before))
        return x
    return calc_time

def remove_space(text):
    return re.sub(' +', ' ', text)


def take(iterable, start, end):
    "Return items (from start to end) of the iterable as a list"
    return list(islice(iterable, start, end))


def convert_to_concepts(image_name: str, dataset_name='V3C') -> list:
    '''
    Function to convert a string into the concepts having the detail of each image in VBS dataset
    
    params:
        - image_name: str
            Input image's name
        - dataset_name: str, default='V3C'
            The name of dataset to convert the concept
    
    return:
       - _: dict
           Detail information related to the dataset
    '''
    concepts = {}
    name = image_name.split('/')[-1].split('.')[0]
    components = name.split('_')
    if dataset_name == 'V3C1' or dataset_name == 'V3C':
        # dataset = None
        video = components[-3][4:]
        shot = components[-2]
        # concepts = {'path': image_name, 'filename': name, 'dataset': dataset_name, 'video': video, 'shot': shot}
    elif dataset_name == 'marine':
        # name = image_name.split('/')[-1]
        # components = name.split('_')
        # dataset = None
        video = '_'.join(components[:-1])
        shot = components[-1].split('.')[0]
    concepts = {'path': image_name, 'filename': name, 'dataset': dataset_name, 'video': video, 'shot': shot}
    return concepts