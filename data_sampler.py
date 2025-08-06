import json
import os
import random
import shutil

folder = './data/omni3d-bench'
image_folder = os.path.join(folder, 'images')
result_folder = "./data/omni3d-samples"
os.mkdir(result_folder)
result_image_folder = os.path.join(result_folder, 'images')

json_path = os.path.join(folder, 'annotations.json')
with open(json_path, 'r') as f:
    json_file = json.load(f)

random.seed(30)
sampled_data = random.sample(json_file['questions'], 20)

for entry in sampled_data:
    rel_path = entry["image_filename"]  # full path
    src_path = os.path.join(image_folder, rel_path)
    dst_path = os.path.join(result_image_folder, rel_path)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)

result_json_path = os.path.join(result_folder, 'annotations.json')
with open(result_json_path, 'w') as f:
    json.dump({"questions": sampled_data}, f, indent=2)