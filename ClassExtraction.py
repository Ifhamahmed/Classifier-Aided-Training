import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm

# import colour palette
df = pd.read_csv('classes.csv', ",", header=None)
palette = np.array(df.values, dtype=np.uint8)
num_of_Classes = palette.shape[0]

# Class names in order of classes in classes.csv file
names = ['unlabeled', 'building', 'sky', 'vegetation', 'pole', 'traffic sign', 'traffic light', 'bus', 'car',
         'person', 'road', 'sidewalk', 'terrain', 'dynamic', 'bicycle', 'ground', 'wall', 'fence', 'rider',
         'motorcycle', 'bridge', 'parking', 'truck', 'train', 'caravan', 'trailer', 'guard rail', 'rail track',
         'tunnel']

frame_path = '/home/ifham/PycharmProjects/gtCoarse/All_frames'
vector_path = '/home/ifham/PycharmProjects/gtCoarse/All_vectors'
print("[INFO] Extracting Class Vectors.................")
for frame in tqdm(os.listdir(frame_path)):
    frm = cv2.imread(frame_path + '/' + frame)
    frm = cv2.resize(frm, (256, 256), interpolation=cv2.INTER_NEAREST)
    class_vector = np.zeros(num_of_Classes, dtype=np.int)
    for i in range(0, frm.shape[0]):
        for j in range(0, frm.shape[1]):
            label = [frm[i, j, 0], frm[i, j, 1], frm[i, j, 2]]
            if label in palette:
                index = np.where((label == palette).all(axis=1))
                class_vector[index] = 1
    vector_name = frame[0:-4]
    np.savetxt(vector_path + '/' + vector_name + '.csv', class_vector, delimiter=',')
print("[INFO] Process Finished..........")




