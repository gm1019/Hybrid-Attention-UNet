# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:33:52 2022

@author: Gong Ming

attention unet

"""




import tensorflow as tf

from keras import backend as K
from keras.callbacks import ModelCheckpoint

from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Cropping2D,
    Dense,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    Lambda,
    MaxPooling2D,
    Permute,
    Reshape,
    UpSampling2D,
    ZeroPadding2D,
    concatenate,
    multiply,
)
from keras.models import Model
from keras.optimizers import Adam

from utils.losses import bce_dice_loss




tf.compat.v1.enable_eager_execution()

#K.clear_session()



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

def spatial_attention(input_feature):
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
		
	return multiply([input_feature, cbam_feature])




def crop(input_x):
    

    
    idx1 = tf.range(input_x.shape[2],0,-1,tf.float32)
    tmp1 = input_x[:,:,:,0]*idx1
    indices1 = tf.argmax(tmp1,-1)
    mask1 = tf.greater(indices1,0)
    non_zero1 = tf.boolean_mask(indices1,mask1)
    left = tf.reduce_min(non_zero1)
    left = tf.cast(left, tf.float32)
    
    indices2 = tf.argmax(tmp1,1)
    mask2 = tf.greater(indices2,0)
    non_zero2 = tf.boolean_mask(indices2,mask2)
    up = tf.reduce_min(non_zero2)
    up = tf.cast(up,tf.float32)
    
    idx2 = tf.range(1,input_x.shape[2]+1,dtype=tf.float32)
    tmp2 = input_x[:,:,:,0]*idx2
    right = tf.reduce_max(tmp2)
    
    idx3 = tf.expand_dims(idx2,axis=1)
    tmp3 = idx3*input_x[:,:,:,0]
    down = tf.reduce_max(tmp3)
    
    x = tf.round(tf.cast((right-left)/2+left,dtype=tf.int32))
    y = tf.round(tf.cast((down-up)/2+up,dtype=tf.int32))
    
    tt = tf.constant(168)
    
    tx = tf.cond(x<tt, lambda:x, lambda:tt)
    ty = tf.cond(y<tt, lambda:y, lambda:tt)
    
    
    tl=tf.constant(90)
    # in_size=tf.constant(512)
    
    w_off = tf.cond(tx>tl, lambda:tx-tl, lambda:40)
    h_off = tf.cond(ty>tl, lambda:ty-tl, lambda:40)
    w_end = tf.add(w_off,tf.constant(176))
    h_end = tf.add(h_off,tf.constant(176))
    
    
    
    
    xatt_cropped = input_x[:, h_off: h_end, w_off: w_end]
    xatt_cropped = tf.image.resize(xatt_cropped,[176,176])


    return xatt_cropped


def threshold(x):
    y = tf.round(x)
    return y


def restore(input_x):

    idx1 = tf.range(input_x.shape[2],0,-1,tf.float32)
    tmp1 = input_x[:,:,:,0]*idx1
    indices1 = tf.argmax(tmp1,-1)
    mask1 = tf.greater(indices1,0)
    non_zero1 = tf.boolean_mask(indices1,mask1)
    left = tf.reduce_min(non_zero1)
    left = tf.cast(left, tf.float32)
    
    indices2 = tf.argmax(tmp1,1)
    mask2 = tf.greater(indices2,0)
    non_zero2 = tf.boolean_mask(indices2,mask2)
    up = tf.reduce_min(non_zero2)
    up = tf.cast(up,tf.float32)
    
    idx2 = tf.range(1,input_x.shape[2]+1,dtype=tf.float32)
    tmp2 = input_x[:,:,:,0]*idx2
    
    right = tf.reduce_max(tmp2)
    
    idx3 = tf.expand_dims(idx2,axis=1)
    tmp3 = idx3*input_x[:,:,:,0]
    down = tf.reduce_max(tmp3)
    
    x = tf.round(tf.cast((right-left)/2+left,dtype=tf.int32))
    y = tf.round(tf.cast((down-up)/2+up,dtype=tf.int32))
    
    tt = tf.constant(168)
    
    tx = tf.cond(x<tt, lambda:x, lambda:tt)
    ty = tf.cond(y<tt, lambda:y, lambda:tt)
    
    
    tl=tf.constant(90)
    # in_size=tf.constant(512)
    
    w_off = tf.cond(tx>tl, lambda:tx-tl, lambda:40)
    h_off = tf.cond(ty>tl, lambda:ty-tl, lambda:40)
    w_end = tf.add(w_off,tf.constant(176))
    h_end = tf.add(h_off,tf.constant(176))
    
    inputx = Cropping2D(cropping=((40, 40), (40, 40)))(tf.expand_dims(input_x[:,:,:,1],-1))
    paddings=[[0,0],[h_off,80-h_off],[w_off,80-w_off],[0,0]]
    #paddings1=[[0,0],[h_off,256-h_off],[128,128],[0,0]]
    out = tf.pad(inputx,paddings,mode="CONSTANT")
    out = tf.image.resize(out,[256,256])
    
    return out





def conv(inputs,filters):
    conv1 = Conv2D(filters, 3, use_bias=False, padding = 'same', kernel_initializer='glorot_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    act1 = Activation("relu")(conv1)
    
    conv2 = Conv2D(filters, 3, use_bias=False, padding = 'same', kernel_initializer='glorot_normal')(act1)
    conv2 = BatchNormalization()(conv2)
    act2 = Activation("relu")(conv2)

    
    return act2



def resd(x1,x2,filters):
    extra = Conv2D(filters, 1, padding="same", activation='relu')(x2)
    ext = Add()([x1,extra])
    
    return ext



def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)


    
    conv1 = conv(inputs,filters=32)
    extra1 = Conv2D(32, 1, padding="same", activation='relu')(inputs)
    ext1 = Add()([conv1,extra1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(ext1)
    #drop1 = Dropout(0.3)(pool1)
    
    conv2 = conv(pool1,filters=64)
    extra2 = Conv2D(64,1,padding="same",activation='relu')(pool1)
    ext2 = Add()([conv2,extra2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(ext2)
    #drop2 = Dropout(0.3)(pool2)
    
    
    conv3 = conv(pool2,filters=128)
    extra3 = Conv2D(128,1,padding="same",activation='relu')(pool2)
    ext3 = Add()([conv3,extra3])
    pool3 = MaxPooling2D(pool_size=(2, 2))(ext3)
    #drop3 = Dropout(0.3)(pool3)
    
    conv4 = conv(pool3,filters=256)
    extra4 = Conv2D(256,1,padding="same",activation='relu')(pool3)
    ext4 = Add()([conv4,extra4])
    pool4 = MaxPooling2D(pool_size=(2, 2))(ext4)
    
    
    conv5 = conv(pool4,filters=256)
    
    
    up6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([ext4,up6], axis = 3)
    conv6 = conv(merge6,filters=256)

    up7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([ext3,up7], axis = 3)
    conv7 = conv(merge7,filters=128)
    
    up8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([ext2,up8], axis = 3)
    conv8 = conv(merge8,filters=64)

    up9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([ext1,up9], axis = 3)
    conv9 = conv(merge9,filters=32)
    
    
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    

    mask = Lambda(threshold)(conv10)
    #################################################################################################################################

    
    
    bd = concatenate([mask,inputs], axis = 3)
    # x1 = tf.constant([0])
    # bd = Add()([boundary,x1],axis=0)
    #boundary = tf.constant([2,3,4,5])
    
    inputs2 = Lambda(crop)(bd)
    #inputs2 = Cropping2D(cropping=((64,64),(64,64)), input_shape=input_size)(inputs)

    n_conv1 = conv(inputs2,filters=128)
    n_conv1 = cbam_block(n_conv1)
    n_ext1 = resd(n_conv1,inputs2,filters=128)
    n_pool1 = MaxPooling2D(pool_size=(2, 2))(n_ext1)
    #drop1 = Dropout(0.3)(pool1)
    
    n_conv2 = conv(n_pool1,filters=256)
    n_conv2 = cbam_block(n_conv2)
    n_ext2 = resd(n_conv2,n_pool1,filters=256)
    n_pool2 = MaxPooling2D(pool_size=(2, 2))(n_ext2)
    #drop2 = Dropout(0.3)(pool2)
    
    
    n_conv3 = conv(n_pool2,filters=512)
    n_conv3 = cbam_block(n_conv3)
    n_ext3 = resd(n_conv3,n_pool2,filters=512)
    n_pool3 = MaxPooling2D(pool_size=(2, 2))(n_ext3)
    #drop3 = Dropout(0.3)(pool3)
    

    n_conv5 = conv(n_pool3,filters=512)
    n_conv5 = cbam_block(n_conv5)
    #conv51 = conv(pool4,filters=512)

    n_up6 = Conv2DTranspose(512, 3, strides=(2,2) ,activation = 'relu', padding = 'same',kernel_initializer='glorot_uniform')(n_conv5)
    n_merge6 = concatenate([n_ext3,n_up6], axis = 3)
    n_conv6 = conv(n_merge6,filters=512)
    n_conv6 = cbam_block(n_conv6)
    n_ext6 = resd(n_conv6,n_merge6,filters=512)


    #up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))  
    n_up7 = Conv2DTranspose(256, 3, strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(n_ext6)
    n_merge7 = concatenate([n_ext2,n_up7], axis = 3)
    n_conv7 = conv(n_merge7,filters=256)
    n_conv7 = cbam_block(n_conv7)
    n_ext7 = resd(n_conv7,n_merge7,filters=256)
    
    #up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    n_up8 = Conv2DTranspose(128, 3, strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(n_ext7)
    n_merge8 = concatenate([n_ext1,n_up8], axis = 3)
    n_conv8 = conv(n_merge8,filters=128)
    n_conv8 = cbam_block(n_conv8)
    n_ext8 = resd(n_conv8,n_merge8,filters=128)
    
    
    n_up9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(n_ext8)

    n_conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(n_up9)

    #n_conv11 = ZeroPadding2D(padding=((64,64),(64,64)), data_format=None)(n_conv9)
    #n_conv11 = Restore()([mask,n_conv9])
    

    n_conv10 = Conv2D(1, 1, activation = 'sigmoid')(n_conv9)
    
    
    
    
    
    m1 = ZeroPadding2D(padding=((40, 40),(40,40)), data_format=None)(n_conv10)
    #m1 = Conv2DTranspose(1, 3, strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(n_conv10)
    

    m_conv10 = concatenate([mask, m1], axis=3)

    n_conv11 = Lambda(restore)(m_conv10)
    
    
    out = Conv2D(1, 1, activation = 'sigmoid')(n_conv11)

    
    model = Model(input = inputs, output = [conv10, n_conv11])


    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-4), loss = [bce_dice_loss,bce_dice_loss], metrics = ['accuracy'])
    
      

    #conv10 = Conv2D(1, 1, activation = 'relu')(conv9)



    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    
    model.summary()

    if(pretrained_weights):
     	model.load_weights(pretrained_weights)

    return model







