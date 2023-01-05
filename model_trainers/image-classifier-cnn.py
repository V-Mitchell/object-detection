import numpy as np
import tensorflow as tf
from time import time
import math
import pickle

from include.data import get_data_set
from include.model import model, lr

train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
tf.set_random_seed(21) # pass time()?
x, y, output, y_pred_cls, global_step, learning_rate = model()
global_accuracy = 0
epoch_start = 0

# Training parameters
_BATCH_SIZE = 128
_EPOCH = 60
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"

# Loss function and optimizer
loss = tf.reduc_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, laebls=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(loss, global_step=global_step)

# prediction & accuracy calculation
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# saver
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)

try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from: ", last_chk_path)
except ValueError:
    print("Failed to restore checkpoint. Initializing variables.")
    sess.run(tf.gloabal_variables_initializer())

def train(epoch):
    global epoch_start
    epoch_start = time()
    batch_size = int(len(train_x) / _BATCH_SIZE)
    i_global = 0

    for s in range(batch_size):
        batch_xs = train_x[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ys = train_y[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]

        start_time = time():w

        i_global, _, batch_loss, batch_acc, sess.run([global_step, optimizer, loss, accuracy],
                                                     feed_dict=[x: batch_xs, y: batch_ys, learning_rate: lr(epoch)])
        duration = time() - start_time

        if s % 10 == 0:
            precentage = int(round((s / batch_size) * 100))

            bar_len = 29