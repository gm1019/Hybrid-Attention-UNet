# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 19:21:46 2026

@author: Ming Gong
"""

from keras.models import load_model

from dataset.data import testGenerator, saveResult
from utils.losses import bce_dice_loss, dice_coef, dice_loss, focal_loss


model = load_model(
    "checkpoints/hybrid_attention_unet.h5",
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coef": dice_coef,
        "dice_loss": dice_loss,
        "focal_loss": focal_loss,
    },
)

test_generator = testGenerator(
    "data/membrane/test_tumour/patient1/patient1",
    num_image=81,
)

liver_predictions, tumour_predictions = model.predict(
    test_generator,
    steps=81,
    verbose=1,
)

saveResult(
    "results/patient1/liver",
    liver_predictions,
)

saveResult(
    "results/patient1/tumour",
    tumour_predictions,
)