import os
import json
import random

# file_path = '/mnt/workspace/cc_sbu_align/filter_cap.json'
file_path = '/mnt/workspace/InstructionGPT-4/filter_cap.json'

# read json
with open(file_path, 'rb') as f:
    data = json.load(f)

anno = data['annotations']

llava_data = []

prompts = [
    "Describe this image in detail",
    "Could you describe the contents of this image for me?",
    "I'd like a detailed description of what's in this image",
    "Can you provide a thorough description of this picture?",
    "Please describe the elements captured in this image",
    "I need a comprehensive description of this picture",
    "Could you paint a detailed picture of what's happening here?",
    "Describe the scene depicted in this image in detail",
    "Can you give me a thorough rundown of what's visible in this image?",
    "I'd appreciate a detailed description of the contents of this image",
    "Please provide a comprehensive account of what's shown in this picture",
    "Could you thoroughly describe the details captured in this image?"
]

for item in anno:
    new_item = {}
    id = item['image_id']
    ans = item['caption']
    new_item['id'] = id
    new_item['image'] = os.path.join('/mnt/workspace/cc_sbu_align/image',id+'.jpg')
    conv = []
    query = random.choice(prompts)
    conv.append(
        {
            "from": "human",
            "value": "<image>\n"+query
        }
    )
    conv.append(
        {
            "from": "gpt",
            "value": ans
        }
    )
    new_item['conversations'] = conv
    llava_data.append(new_item)

with open('/mnt/workspace/instruct-4_data.json','w') as f:
    json.dump(llava_data, f)
