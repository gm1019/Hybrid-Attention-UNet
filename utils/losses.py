# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 18:32:30 2026

@author: Ming Gong
"""


from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import keras.backend as K
import keras
import tensorflow as tf
from keras import losses





def focal_loss(y_true, y_pred):
          
    alpha = tf.constant(2, dtype=tf.float32)
    gamma = tf.constant(0.25, dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
    p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
    loss = K.mean(focal_loss)
    
    return loss


def focal_loss1(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed



def bce_focal_loss(y_true,y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + focal_loss(y_true, y_pred)
    return loss


def bce_focal_loss1(y_true,y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + focal_loss1(y_true, y_pred)
    return loss
    



def tversky(y_true, y_pred):
    smooth = 1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)




def dice_coef(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss