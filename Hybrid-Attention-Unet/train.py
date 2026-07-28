# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 18:51:51 2026

@author: Ming Gong
"""


import tensorflow as tf
from models.euvip import *
from dataset.data import trainGenerator, testGenerator, saveResult
import cv2
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
import os
import numpy as np
import glob
from utils.losses import dice_coef,bce_dice_loss,dice_loss,focal_loss

tf.compat.v1.enable_eager_execution()
K.clear_session()


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

#reduce_lr = ReduceLROnPlateau(monitor='loss',factor=0.1, patience=10, mode='auto')


train_generator = trainGenerator(
    batch_size=3,
    train_path="data/membrane/train",
    image_folder="image_t",
    mask_folder="label_tl",
    maskt_folder="label_t",
    aug_dict=data_gen_args,
    save_to_dir=None,
)

model = unet()

checkpoint = ModelCheckpoint(
    "checkpoints/hybrid_attention_unet.h5",
    monitor="loss",
    verbose=1,
    save_best_only=True,
)

model.fit(
    train_generator,
    steps_per_epoch=1600,
    epochs=20,
    callbacks=[checkpoint],
)

model.save("checkpoints/final_model.h5")