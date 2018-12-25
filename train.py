import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
from models import generator, discriminator
from losses import l1_loss, l2_loss, smooth_loss
from data import load_data
import time


# ============= hyper-parameters =============
face_dir = 'data/imgs/'
au_dir = 'data/aus.pkl'

BATCH_SIZE = 25
EPOCHS = 30

lambda_D_img = 1
lambda_D_au = 4000
lambda_D_gp = 10
lambda_cyc = 10
lambda_mask = 0.1
lambda_mask_smooth = 1e-5


# ============= placeholder =============
real_img = tf.placeholder(tf.float32, [None, 128, 128, 3], name='real_img')
real_au = tf.placeholder(tf.float32, [None, 17], name='real_au')
desired_au = tf.placeholder(tf.float32, [None, 17], name='desired_au')
lr = tf.placeholder(tf.float32, name='lr')


# ============= G & D =============
# G(Ic1, c2) * M
fake_img, fake_mask = generator(real_img, desired_au, reuse=False)
fake_img_masked = fake_mask * real_img + (1 - fake_mask) * fake_img
# G(G(Ic1, c2)*M, c1) * M
cyc_img, cyc_mask = generator(fake_img_masked, real_au, reuse=True)
cyc_img_masked = cyc_mask * fake_img_masked + (1 - cyc_mask) * cyc_img

# D(real_I)
pred_real_img, pred_real_au = discriminator(real_img, reuse=False)
# D(fake_I)
pred_fake_img_masked, pred_fake_au = discriminator(fake_img_masked, reuse=True)


# ============= losses =============
# G losses
loss_g_fake_img_masked = -tf.reduce_mean(pred_fake_img_masked) * lambda_D_img
loss_g_fake_au = l2_loss(desired_au, pred_fake_au) * lambda_D_au
loss_g_cyc = l1_loss(real_img, cyc_img_masked) * lambda_cyc

loss_g_mask_fake = tf.reduce_mean(fake_mask) * lambda_mask + smooth_loss(fake_mask) * lambda_mask_smooth
loss_g_mask_cyc = tf.reduce_mean(cyc_mask) * lambda_mask + smooth_loss(cyc_mask) * lambda_mask_smooth

loss_g = loss_g_fake_img_masked + loss_g_fake_au + \
         loss_g_cyc + \
         loss_g_mask_fake + loss_g_mask_cyc

# D losses
loss_d_img = -tf.reduce_mean(pred_real_img) * lambda_D_img + tf.reduce_mean(pred_fake_img_masked) * lambda_D_img
loss_d_au = l2_loss(real_au, pred_real_au) * lambda_D_au

alpha = tf.random_uniform([BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.)
differences = fake_img_masked - real_img
interpolates = real_img + tf.multiply(alpha, differences)
gradients = tf.gradients(discriminator(interpolates, reuse=True), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
loss_d_gp = lambda_D_gp * gradient_penalty

loss_d = loss_d_img + loss_d_au + loss_d_gp


# ============= summary =============
real_img_sum = tf.summary.image('real_img', real_img)

fake_img_sum = tf.summary.image('fake_img', fake_img)
fake_mask_sum = tf.summary.image('fake_mask', fake_mask)
fake_img_masked_sum = tf.summary.image('fake_img_masked', fake_img_masked)

cyc_img_sum = tf.summary.image('cyc_img', cyc_img)
cyc_mask_sum = tf.summary.image('cyc_mask', cyc_mask)
cyc_img_masked_sum = tf.summary.image('cyc_img_masked', cyc_img_masked)

loss_g_sum = tf.summary.scalar('loss_g', loss_g)
loss_g_fake_img_masked_sum = tf.summary.scalar('loss_g_fake_img_masked', loss_g_fake_img_masked)
loss_g_fake_au_sum = tf.summary.scalar('loss_g_fake_au', loss_g_fake_au)
loss_g_cyc_sum = tf.summary.scalar('loss_g_cyc', loss_g_cyc)
loss_g_mask_fake_sum = tf.summary.scalar('loss_g_mask_fake', loss_g_mask_fake)
loss_g_mask_cyc_sum = tf.summary.scalar('loss_g_mask_cyc', loss_g_mask_cyc)

loss_d_sum = tf.summary.scalar('loss_d', loss_d)
loss_d_img_sum = tf.summary.scalar('loss_d_img', loss_d_img)
loss_d_au_sum = tf.summary.scalar('loss_d_au', loss_d_au)
loss_d_gp_sum = tf.summary.scalar('loss_d_gp', loss_d_gp)

g_sum = tf.summary.merge([loss_g_sum, loss_g_fake_img_masked_sum, loss_g_fake_au_sum, loss_g_cyc_sum, loss_g_mask_fake_sum, loss_g_mask_cyc_sum, 
                          real_img_sum,
                          fake_img_sum, fake_mask_sum, fake_img_masked_sum,
                          cyc_img_sum, cyc_mask_sum, cyc_img_masked_sum])

d_sum = tf.summary.merge([loss_d_sum, loss_d_img_sum, loss_d_au_sum, loss_d_gp_sum])


# ============= load data =============
face, au = load_data(face_dir, au_dir)
au_rand = au.copy()
np.random.shuffle(au_rand)
au_rand += np.random.uniform(-0.1, 0.1, au_rand.shape)


# ============= train =============
train_vars = tf.trainable_variables()
g_vars = [var for var in train_vars if var.name.startswith('Generator')]
d_vars = [var for var in train_vars if var.name.startswith('Discriminator')]

g_train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999).minimize(loss_g, var_list=g_vars)
d_train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999).minimize(loss_d, var_list=d_vars)

saver = tf.train.Saver()
d_loss = 0
g_loss = 0
counter = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./logs/", sess.graph)
    print('----------- start training -----------')

    for e in range(1, EPOCHS+1):
        start_time = time.time()
        if e <= 21:
            lr_now = 1e-4
        else:
            lr_now = 1e-5 * (EPOCHS + 1 - e)
        print('===== [Epoch %02d/30](lr: %.5f) =====' % (e, lr_now))

        for i in range(len(face) // BATCH_SIZE):
            d_loss, summary_str, _ = sess.run([loss_d, d_sum, d_train_op],
                                              feed_dict={real_img: face[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                                                         real_au: au[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                                                         desired_au: au_rand[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                                                         lr: lr_now})
            writer.add_summary(summary_str, counter)
            if (i+1) % 5 == 0:
                g_loss, summary_str, _ = sess.run([loss_g, g_sum, g_train_op],
                                                  feed_dict={real_img: face[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                                                             real_au: au[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                                                             desired_au: au_rand[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                                                             lr: lr_now})
                writer.add_summary(summary_str, counter)
            counter += 1

        print('(spend time: %.2fmin) loss_g: %.4f  loss_d: %.4f \n' %
              ((time.time()-start_time)/60, g_loss, d_loss))

        saver.save(sess, 'weights/ganimation_epoch%2d.ckpt' % e)
