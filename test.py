import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
import numpy as np
import cv2
import face_recognition
from models import generator, discriminator
from tqdm import tqdm


real_img = tf.placeholder(tf.float32, [1, 128, 128, 3], name='real_img')
style_img = tf.placeholder(tf.float32, [1, 128, 128, 3], name='style_img')

_, desired_au = discriminator(style_img, reuse=False)
fake_img, fake_mask = generator(real_img, desired_au, reuse=False)
fake_img_masked = fake_mask * real_img + (1 - fake_mask) * fake_img

imgs_names = os.listdir('jiaqi_face')
real_src = face_recognition.load_image_file('obama.jpeg')  # RGB image
face_loc = face_recognition.face_locations(real_src)
if len(face_loc) == 1:
    top, right, bottom, left = face_loc[0]

real_face = np.zeros((1,128,128,3), dtype=np.float32)
style_face = np.zeros((1,128,128,3), dtype=np.float32)
real_face[0] = cv2.resize(real_src[top:bottom, left:right], (128,128)) / 127.5 - 1


saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'weights/ganimation_upsample_epoch39.ckpt')
    for img_name in tqdm(imgs_names):
        style_face[0] = cv2.imread('jiaqi_face/'+img_name)[:,:,::-1] / 127.5 - 1
        #print(sess.run(desired_au, feed_dict={style_img:style_face}))
        output = sess.run(fake_img_masked, feed_dict={real_img:real_face, style_img:style_face})
        real_src[top:bottom, left:right] = cv2.resize((output[0]+1)*127.5, (right-left,bottom-top))
        cv2.imwrite('output/'+img_name, real_src[:,:,::-1])
