import os
import glob
from tqdm import tqdm
import numpy as np
import pickle


face_path = 'imgs/'
aus_path = 'aus_openface/'
aus_save_path = 'aus.pkl'


face_names = os.listdir(face_path)
face_names.sort()

file_paths = glob.glob(os.path.join(aus_path, '*.csv'))
file_paths.sort()

data = dict()
for file_path in tqdm(file_paths):
    file_name = os.path.basename(file_path[:-4])
    if file_name+'.jpg' in face_names:
        content = np.loadtxt(file_path, delimiter=', ', skiprows=1)
        if content.ndim == 1 and len(content) == 37:
            data[file_name] = content[2:19]

with open(aus_save_path, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
