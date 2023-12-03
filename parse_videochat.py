import os
import json

json_path = "/Users/xiaodong/Downloads/videochat_instruct_11k.json"

with open(json_path, 'rb') as f:
    data = json.load(f)

print(len(data))

video_names = []
for item in data:
    id = item['video'].split('/')[1].split('.')[0]
    video_names.append(id)

video_names = list(set(video_names))

# print(video_names)
print(len(video_names))

# write video_names to json
with open('videochat_train_ids.json', 'w') as f:
    json.dump(video_names, f)