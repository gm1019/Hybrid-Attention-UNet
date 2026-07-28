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


myGene = trainGenerator(3,'data/membrane/train','image_t','label_tl','label_t',data_gen_args,save_to_dir = None)



model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.h5', monitor='loss',verbose=1, save_best_only=True)
# early_stopping = EarlyStopping(monitor='loss', patience=5,mode='min')
model.fit_generator(myGene,steps_per_epoch=1600,epochs=20,callbacks=[model_checkpoint])
#model.fit_generator(myGene,steps_per_epoch=2500,epochs=20,callbacks=[reduce_lr])
model.save('data/membrane/predict/my_unet.h5')





testGen1 = testGenerator("data/membrane/test_tumour/patient1/patient1",num_image=81)
testGen2 = testGenerator("data/membrane/test_tumour/patient1/patient2",num_image=26)
testGen3 = testGenerator("data/membrane/test_tumour/patient1/patient3",num_image=28)
testGen4 = testGenerator("data/membrane/test_tumour/patient1/patient4",num_image=41)
testGen5 = testGenerator("data/membrane/test_tumour/patient1/patient5",num_image=240)
testGen6 = testGenerator("data/membrane/test_tumour/patient1/patient6",num_image=76)
testGen7 = testGenerator("data/membrane/test_tumour/patient1/patient7",num_image=38)
testGen8 = testGenerator("data/membrane/test_tumour/patient1/patient8",num_image=74)
testGen9 = testGenerator("data/membrane/test_tumour/patient1/patient9",num_image=34)
testGen10 = testGenerator("data/membrane/test_tumour/patient1/patient10",num_image=41)
testGen11 = testGenerator("data/membrane/test_tumour/patient1/patient11",num_image=33)
testGen12 = testGenerator("data/membrane/test_tumour/patient1/patient12",num_image=52)
testGen13 = testGenerator("data/membrane/test_tumour/patient1/patient13",num_image=20)
testGen14 = testGenerator("data/membrane/test_tumour/patient1/patient14",num_image=17)
testGen15 = testGenerator("data/membrane/test_tumour/patient1/patient15",num_image=91)
testGen16 = testGenerator("data/membrane/test_tumour/patient1/patient16",num_image=64)
testGen17 = testGenerator("data/membrane/test_tumour/patient1/patient17",num_image=51)
testGen18 = testGenerator("data/membrane/test_tumour/patient1/patient18",num_image=77)
testGen19 = testGenerator("data/membrane/test_tumour/patient1/patient19",num_image=29)
testGen20 = testGenerator("data/membrane/test_tumour/patient1/patient20",num_image=80)

middle = Model(inputs=model.input,outputs=model.get_layer('lambda_2').output)
# midle = Model(inputs=model.input,outputs=model.get_layer('lambda_2').output)


results1 = middle.predict_generator(testGen1,81,verbose=1)
results2 = middle.predict_generator(testGen2,26,verbose=1)
results3 = middle.predict_generator(testGen3,28,verbose=1)
results4 = middle.predict_generator(testGen4,41,verbose=1)
results5 = middle.predict_generator(testGen5,240,verbose=1)
results6 = middle.predict_generator(testGen6,76,verbose=1)
results7 = middle.predict_generator(testGen7,38,verbose=1)
results8 = middle.predict_generator(testGen8,74,verbose=1)
results9 = middle.predict_generator(testGen9,34,verbose=1)
results10 = middle.predict_generator(testGen10,41,verbose=1)
results11 = middle.predict_generator(testGen11,33,verbose=1)
results12 = middle.predict_generator(testGen12,52,verbose=1)
results13 = middle.predict_generator(testGen13,20,verbose=1)
results14 = middle.predict_generator(testGen14,17,verbose=1)
results15 = middle.predict_generator(testGen15,91,verbose=1)
results16 = middle.predict_generator(testGen16,64,verbose=1)
results17 = middle.predict_generator(testGen17,51,verbose=1)
results18 = middle.predict_generator(testGen18,77,verbose=1)
results19 = middle.predict_generator(testGen19,29,verbose=1)
results20 = middle.predict_generator(testGen20,80,verbose=1)







def saveResult1(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):

        img=item[:,:,1]
            #print(np.max(img),np.min(img))
        img = img*256
            #print(np.max(img),np.min(img))
            
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)



saveResult1("data/membrane/predict/predict_tumour/predict1",results1)
saveResult1("data/membrane/predict/predict_tumour/predict2",results2)
saveResult1("data/membrane/predict/predict_tumour/predict3",results3)
saveResult1("data/membrane/predict/predict_tumour/predict4",results4)
saveResult1("data/membrane/predict/predict_tumour/predict5",results5)
saveResult1("data/membrane/predict/predict_tumour/predict6",results6)
saveResult1("data/membrane/predict/predict_tumour/predict7",results7)
saveResult1("data/membrane/predict/predict_tumour/predict8",results8)
saveResult1("data/membrane/predict/predict_tumour/predict9",results9)
saveResult1("data/membrane/predict/predict_tumour/predict10",results10)
saveResult1("data/membrane/predict/predict_tumour/predict11",results11)
saveResult1("data/membrane/predict/predict_tumour/predict12",results12)
saveResult1("data/membrane/predict/predict_tumour/predict13",results13)
saveResult1("data/membrane/predict/predict_tumour/predict14",results14)
saveResult1("data/membrane/predict/predict_tumour/predict15",results15)
saveResult1("data/membrane/predict/predict_tumour/predict16",results16)
saveResult1("data/membrane/predict/predict_tumour/predict17",results17)
saveResult1("data/membrane/predict/predict_tumour/predict18",results18)
saveResult1("data/membrane/predict/predict_tumour/predict19",results19)
saveResult1("data/membrane/predict/predict_tumour/predict20",results20)










# t = 1608

# path = "C:/Users/xgb15139/Desktop/unet-master/data/membrane/mask_t" #文件夹目录
# files= os.listdir(path) #得到文件夹下的所有文件名称
# files.sort(key=lambda x:int(x[:-4]))


# images = np.zeros([t,512,512,1])

# for i in range(t):
    
#     curr = cv2.imread(path + '/' + files[i],0)
#     curr = curr.reshape((1,512,512,1))
#     images[i,:,:,:] = curr
    
# Iou = compute_iou(images,results[0:t])   
    
    
# IoU_ave = sum(compute_iou(images,results))/t
    
    





#model.load_weights(os.path.join(model_path, config.exp_name+".h5"))

#p_test = model.predict(x_test, batch_size=config.batch_size, verbose=config.verbose)
#eva = model.evaluate(x_test, y_test, batch_size=config.batch_size, verbose=config.verbose)



#IoU = compute_iou(mask,results)


#print(">> Testing dataset mIoU  = {:.2f}%".format(np.mean(IoU)))
#print(">> Testing dataset mDice = {:.2f}%".format(eva[3]*100.0))






#path = "C:/Users/xgb15139/Desktop/unet-master/data/membrane/mask" #文件夹目录
#a = cv2.imread(path + '/341.jpg',-1)
#b = cv2.imread('C:/Users/xgb15139/Desktop/unet-master/data/membrane/test/341_predict.jpg',-1)
#a = a/255
#b = b/255
#
#a = a.reshape((1,512,512,1))
#b = b.reshape((1,512,512,1))
#score = compute_iou(a,b)
##cv2.imshow('a', a)
#cv2.waitKey(0)

