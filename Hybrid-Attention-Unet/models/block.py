# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:08:33 2022

@author: Gong Ming
"""


import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import losses 



def spatial_attention(input_feature1,input_feature2):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature._keras_shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature._keras_shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat._keras_shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	assert cbam_feature._keras_shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return  multiply([input_feature2, cbam_feature])




def psp_block(inputs, filters, kernel_size):
    
    pool_1 = AveragePooling2D(pool_size=(2, 2))(inputs)
    pool_2 = AveragePooling2D(pool_size=(4, 4))(inputs)
    pool_3 = AveragePooling2D(pool_size=(8, 8))(inputs)
    pool_4 = AveragePooling2D(pool_size=(16,16))(inputs)
    
    conv_1 = Conv2D(256, 3, activation = 'relu', dilation_rate=(2,2), padding = 'same', kernel_initializer = 'glorot_normal')(pool_1)
    conv_1 = Conv2D(filters, 1, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv_1)
    conv_2 = Conv2D(256, 3, activation = 'relu', dilation_rate=(2,2), padding = 'same', kernel_initializer = 'glorot_normal')(pool_2)
    conv_2 = Conv2D(filters, 1, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv_2)
    conv_3 = Conv2D(256, 3, activation = 'relu', dilation_rate=(3,3), padding = 'same', kernel_initializer = 'glorot_normal')(pool_3)
    conv_3 = Conv2D(filters, 1, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv_3)
    conv_4 = Conv2D(256, 3, activation = 'relu', dilation_rate=(3,3), padding = 'same', kernel_initializer = 'glorot_normal')(pool_4)
    conv_4 = Conv2D(filters, 1, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv_4)
    
    
    up_1 = Conv2DTranspose(filters, 3, strides=(2, 2), activation = 'relu', padding='same')(conv_1)
    up_2 = Conv2DTranspose(filters, 3, strides=(4, 4), activation = 'relu', padding='same')(conv_2)
    up_3 = Conv2DTranspose(filters, 3, strides=(8, 8), activation = 'relu', padding='same')(conv_3)
    up_4 = Conv2DTranspose(filters, 3, strides=(16, 16), activation = 'relu', padding='same')(conv_4)
    
    merge = concatenate([inputs,up_1,up_2,up_3,up_4], axis = 3)
    merge = Conv2D(kernel_size, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(merge)
    
    return merge


def se_block(input_feature, ratio=8):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)

	se_feature = multiply([input_feature, se_feature])
	return se_feature


def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def cbam_block(cbam_feature, ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature

def channel_attention(input_feature, ratio=8):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])
