import json
import os
import random
import pickle

property = "scene"  # "object"
data_root = f"/mnt/d/dataset/EmoSet/0103_split_to_folder/{property}"
image_paths = []
for root, _, file_path in os.walk(data_root):
    for file in file_path:
        if file.endswith("jpg"):
            flag = False
            path = os.path.join(root, file)
            emotion = path.split('/')[-1].split('_')[0]
            number = path.split('/')[-1].split('.')[0].split('_')[1]
            attribute = path.split('/')[-2].split(')')[-1].lower().replace(' ', '_')
            annotion_path = f'/mnt/d/dataset/EmoSet/annotation/{emotion}/{emotion}_{number}.json'
            annotion = json.load(open(annotion_path, 'r'))
            if 'scene' in annotion:
                flag = True
                tmp = annotion['scene'].lower().replace(' ', '_')
                if tmp == attribute:
                    image_paths.append(path)
            if flag is False:
                try:
                    tmp = annotion['object'][0].lower().replace(' ', '_')
                    if tmp == attribute:
                        image_paths.append(path)
                except:
                    print("annotation is wrong")

# Calculate the number of samples for each emotion label
emotion_counts = {}
for path in image_paths:
    emotion = path.split('/')[-1].split('_')[0]
    if emotion not in emotion_counts:
        emotion_counts[emotion] = []
    emotion_counts[emotion].append(path)

# Record the maximum number of samples
max_num = max([len(v) for v in emotion_counts.values()])

# Generate a random index list for each emotion label
index_lists = {}
for emotion, paths in emotion_counts.items():
    random_indices = list(range(max_num))
    random.shuffle(random_indices)
    indices = [i % (len(paths)) for i in random_indices]
    index_lists[emotion] = indices

# Extract images based on index list
image_paths = []
for emo, path in emotion_counts.items():
    list = index_lists[emo]
    for indice in list:
        image_paths.append(path[indice])

# 随机打乱图像列表
random.shuffle(image_paths)

with open(f'dataset_balance/{property}_norepeat.pkl', 'wb') as f:
    pickle.dump(image_paths, f)
