import os
import cv2
import face_recognition
from tqdm import tqdm


data_path = 'data/emotionet/'
save_path = 'data/imgs/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

imgs_names = os.listdir(data_path)
for img_name in tqdm(imgs_names):
    img = face_recognition.load_image_file(os.path.join(data_path, img_name))  # RGB image
    face_loc = face_recognition.face_locations(img)
    if len(face_loc) == 1:
        top, right, bottom, left = face_loc[0]
        if (bottom-top > 100) and (right-left > 100):  # skip the low-resolution face
            face = img[top:bottom, left:right, ::-1]  # BGR face
            face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(save_path, img_name), face)
