import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2

from aug_sequences import aug_seq_1
from datasets.kitti.reader import CARS_ONLY
from datasets.kitti.tfdataset import TFDataset
from losses import focal_loss, smooth_L1_masked
from models import create_model
from target_encoders import ResTargets
from utils import allow_growth, create_dir
from velo_encoders import OccupancyCuboid

allow_growth()

velo_encoder = OccupancyCuboid(delta=(0.16, 0.1, 0.16))
target_encoder = ResTargets(shape=(250, 220), keep_factor=0.5, default_class='Car', confidence_thresh=0.05)

train_ds = TFDataset('train', CARS_ONLY, batch_size=2, velo_encoder=velo_encoder, target_encoder=target_encoder,
                     aug_fn=aug_seq_1(), shuffle=True, prefetch=4, num_parallel_calls=2)

with tf.distribute.MirroredStrategy().scope():
    outputs_dir = create_dir('outputs')
    ckpts_dir = create_dir(os.path.join(outputs_dir, 'ckpts'))
    callbacks = [ModelCheckpoint(os.path.join(ckpts_dir, 'E{epoch:04d}.h5'), save_best_only=False, period=20, verbose=True, save_weights_only=True),
                 TensorBoard(log_dir=os.path.join(outputs_dir, 'logs'))]

    model = create_model((500, 440, 40), C=64, is_train=True, ki='glorot_normal', kr=l2(1e-4))
    model.compile(optimizer=Adam(learning_rate=ExponentialDecay(0.0002, decay_steps=15 * (3712 / 2), decay_rate=0.8, staircase=True)),
                  loss=[focal_loss(alpha=0.75, gamma=1.0), smooth_L1_masked(1 / 9.0), smooth_L1_masked(1 / 9.0), smooth_L1_masked(1 / 9.0)],
                  loss_weights=[1.0, 1.0, 1.0, 1.0])
    model.fit(train_ds, epochs=200, callbacks=callbacks)
