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





def adjustData(img,mask,mask1,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        new_mask1 = np.zeros(mask1.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
            new_mask1[mask1 == i,i]=1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
        new_mask1 = np.reshape(new_mask1,(new_mask1.shape[0],new_mask1.shape[1]*new_mask1.shape[2],new_mask1.shape[3])) if flag_multi_class else np.reshape(new_mask1,(new_mask1.shape[0]*new_mask1.shape[1],new_mask1.shape[2]))
        mask1 = new_mask1
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask1 = mask1 /255
        mask1[mask1 > 0.5] = 1
        mask1[mask1 <= 0.5] = 0
        
        
    return (img,mask,mask1)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,maskt_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",maskt_save_prefix = "maskt",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    maskt_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    maskt_generator = maskt_datagen.flow_from_directory(
        train_path,
        classes = [maskt_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = maskt_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator, maskt_generator)
    for (img,mask,maskt) in train_generator:
        img,mask,maskt = adjustData(img,mask,maskt,flag_multi_class,num_class)
        yield (img,[mask,maskt])


             


def geneTrainNpy(image_path,mask_path,mask1_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True,mask1_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.jpg"%image_prefix))
    image_arr = []
    mask_arr = []
    mask1_arr=[]
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        mask1 = io.imread(item,as_gray = image_as_gray)
        mask1 = np.reshape(mask1,mask1.shape + (1,)) if mask1_as_gray else mask1
        img,mask,mask1 = adjustData(img,mask,mask1,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
        mask1_arr.append(mask1)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    mask1_arr = np.array(mask1_arr)
    return image_arr,mask_arr,mask1_arr






def testGenerator(test_path,num_image,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
       
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)

        yield img


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)







def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 80:
        lr *= 0.5e-4
    elif epoch > 60:
        lr *= 1e-4
    elif epoch > 40:
        lr *= 5e-4
    elif epoch > 20:
        lr *= 1e-3
    print('Learning rate: ', lr)
    return lr

    
    

def compute_iou(img1, img2):

    img1 = np.array(img1)
    
    img2 = np.array(img2)

    if img1.shape[0] != img2.shape[0]:
        raise ValueError("Shape mismatch: the number of images mismatch.")
    IoU = np.zeros( (img1.shape[0],), dtype=np.float32)
    for i in range(img1.shape[0]):
        im1 = np.squeeze(img1[i]>0.5)
        im2 = np.squeeze(img2[i]>0.5)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        if im1.sum() + im2.sum() == 0:
            IoU[i] = 100
        else:
            IoU[i] = (2. * intersection.sum() * 100.0) / (im1.sum() + im2.sum()) 
            
        #database.display_image_mask_pairs(im1, im2)

    return IoU





def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):

        img=item[:,:,0]
           #print(np.max(img),np.min(img))
        img[img>0.5]=1#此时1是浮点数，下面的0也是
        img[img<=0.5]=0
            #print(np.max(img),np.min(img))
            
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        
