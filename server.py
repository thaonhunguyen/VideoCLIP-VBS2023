import sys
from dotenv import load_dotenv
load_dotenv()
from dataset_helpers import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import time
from configs import *

generate_features = True

print("Dataset name: ", DATASET_NAME, file=sys.stdout)
clip = CLIPSearchEngine(src_path=DATASET_MASTER_PATH, feature_path=FEATURE_PATH, generate_features=generate_features)
clip.dataset.get_file_name()
print('Loading Features', file=sys.stdout)
clip.load_features()

# with open("feature_dict.pkl", "rb") as a_file:
#     print('Loading Feature Dict', file=sys.stdout)
#     clip.feature_dict = pickle.load(a_file)
#     a_file.close()

from flask import Flask, request

app = Flask(__name__)

def get_from_obj(key, obj, default_value):
    return obj[key] if key in obj else default_value

def get_from_body(body):
    return lambda key, default_value: get_from_obj(key, body, default_value)

def format_result(result_entity):
    """
    Format result to get just video id and shot id
    """
    return {
        # "path": result_entity['path'],
        "video": result_entity['video'],
        "shot": result_entity['shot'],
        # "id": result_entity['filename'],
    }

def group_result(result):
    """
    TODO: change from one group for one video to 3 groups for one video
    """
    videos = dict()
    results = []
    for item in result:
        video_id = item['video']
        score = item['score']
        if video_id not in videos:
            videos[video_id] = [(score, item)]
        else:
            videos[video_id].append((score, item))
    for key in videos:
        # Sort with in the same video
        videos[key] = sorted(videos[key], key=lambda x: x[0], reverse=True)
        top_3 = videos[key][:3]
        score = sum(i[0] for i in top_3)
        results.append((score, {
            "video": key,
            "keyframes": list(map(format_result, list(map(lambda x: x[1], top_3)))),
            "score": score,
        }))
    # Sort among videos
    results = sorted(results, key=lambda x: x[0], reverse=True)
    return list(map(lambda x: x[1], results))

@app.route('/api/search', methods=['POST'])
def search():
    body = request.get_json()
    body_value = get_from_body(body)
    query = body_value('query', '')
    ocr = body_value('ocr', '')
    colors = body_value('colors', [])
    metas = body_value('metas', [])
    dataset = body_value('dataset', 'V3C')
    total = body_value('total', 100)
    best_images = clip.search_query(query, num_matches=total * 3, ss_type='other')

    # print(best_images[:1])
    return {
        # "result": list(map(format_result, best_images)),
        "result": group_result(best_images)
    }


@app.route('/api/server-time', methods=['GET'])
def servertime():
    current_time = time.time()
    return {
        'current_time': str(current_time),
    }


@app.route('/api/find_similar_keyframes/<video_id>/<keyframe_id>', methods=['GET'])
def similar_keyframes(video_id, keyframe_id):
    # TODO
    img_query = f'shot{video_id}_{keyframe_id}_RKF.png'
    feature = clip.feature_dict[img_query]
    feature_vec = np.expand_dims(feature, axis=0)
    feature_vector = feature_vec.astype('float32')
    similarities = (feature_vector @ clip.features.T).squeeze(0)
    indices = similarities.argsort()[-50:][::-1]
    best_matched_image_names = [clip.dataset.image_names[item] for item in indices]

    result = [convert_to_concepts(item, dataset_name=clip.dataset_name) for item in best_matched_image_names]
    return {
        "result": list(map(format_result, result[:1000])),
    }
    
    
if __name__ == '__main__':
    host = os.environ['HOST']
    port = os.environ['PORT']
    print(f'Running server on {host} port {port}', file=sys.stdout)
    app.run(host=host, port=port, debug=False)
