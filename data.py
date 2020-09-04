import os,math
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
import cv2
#import Augmentor
data_base='F:/data/mars/cancer/'

def getData(data_base):
    flist1=os.listdir(data_base+'0/')
    flist1=pd.DataFrame([data_base+'0/'+ x for x in flist1],columns=['filepath'])
    flist1['maskpath']=''
    flist1['label']=0
    flist2=os.listdir(data_base+'1/')
    flist2=[x for x in flist2 if x[-9:]=='_mask.jpg' ]
    flist2=pd.DataFrame([data_base+'1/'+ x for x in flist2],columns=['maskpath'])
    flist2['filepath']=[x.replace('_mask','') for x in flist2.maskpath]
    flist2['label']=1
    df=pd.concat([flist1[['filepath','label','maskpath']],flist2[['filepath','label','maskpath']]])
    df=df.sample(frac=1)
    ns=int(df.shape[0]*0.8)
    df['type']='train'
    df['type'].values[ns:]='test'
    print(df.groupby(['type','label'])['filepath'].count())
    flist3 = os.listdir(data_base + 'test/')
    test = pd.DataFrame([data_base + 'test/' + x for x in flist3], columns=['filepath'])
    test['filename'] = flist3
    return df,test


def preprocess_input(x):
    if x.dtype not in ['float32', 'float64', 'float']:
        x = x.astype(np.float32)
    x /= 127.5
    x -= 1.
    return x

def randomCrop(img, crop_shape=(224,224),size=256 ):
    img=cv2.resize(img,(size,size))
    width, height=crop_shape
    assert img.shape[0] >= width
    assert img.shape[1] >= height
    #assert img.shape[0] == mask.shape[0]
    #assert img.shape[1] == mask.shape[1]
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    #mask = mask[y:y+height, x:x+width]
    return img

def randomCropMask(img,mask, crop_shape=(224,224),size=256 ):
    img=cv2.resize(img,(size,size))
    mask = cv2.resize(mask, (size, size))
    width, height=crop_shape
    assert img.shape[1] >= height
    #assert img.shape[0] == mask.shape[0]
    #assert img.shape[1] == mask.shape[1]
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img,mask

class DataGenerator(keras.utils.Sequence):
#class DataGenerator:
    def __init__(self,data, batch_size=32,size=96, shuffle=True):
        self.batch_size = batch_size
        self.size=size
        self.data=data
        self.shuffle = shuffle
        self.samples_num=self.data.shape[0]
        self.data.index = [i for i in range(0, self.samples_num)]

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(self.samples_num / float(self.batch_size))

    def __getitem__(self, index):
        i=index*self.batch_size
        length=min(self.batch_size,(self.samples_num-i))
        batch_inputs=np.zeros((length,self.size,self.size,3),dtype=np.float32)
        batch_mask = np.zeros((length, self.size, self.size, 1), dtype=np.float32)
        target=np.zeros((length),dtype=np.float32)
        for i_batch in range(0,length):
            sample=self.data.iloc[i+i_batch].to_dict()
            target[i_batch]=sample['label']
            image=cv2.imread(sample['filepath'])
            image = cv2.resize(image, (self.size, self.size), cv2.INTER_CUBIC)
            image = image[:, :, ::-1]
            batch_inputs[i_batch] = preprocess_input(image)
            if sample['label']==1:
                mask = cv2.imread(sample['maskpath'],0)
                mask[mask > 127] = 255
                mask[mask < 127] = 0
                mask = cv2.resize(mask, (self.size, self.size), cv2.INTER_CUBIC)
                batch_mask[i_batch] = mask.reshape((self.size,self.size,1))/255.
        return (batch_inputs ,[target,batch_mask])
    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            self.data=self.data.sample(frac=1)
            self.data.index=[i for i in range(0,self.samples_num)]

class DataGeneratorC(keras.utils.Sequence):
#class DataGenerator:
    def __init__(self,data, batch_size=32,size=96, shuffle=True,au=False):
        self.batch_size = batch_size
        self.size=size
        self.data=data
        self.shuffle = shuffle
        self.au=au
        self.samples_num=self.data.shape[0]
        self.data.index = [i for i in range(0, self.samples_num)]

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(self.samples_num / float(self.batch_size))

    def __getitem__(self, index):
        i=index*self.batch_size
        length=min(self.batch_size,(self.samples_num-i))
        batch_inputs=np.zeros((length,self.size,self.size,3),dtype=np.float32)
        #batch_mask = np.zeros((length, self.size, self.size, 1), dtype=np.float32)
        target=np.zeros((length),dtype=np.float32)
        for i_batch in range(0,length):
            sample=self.data.iloc[i+i_batch].to_dict()
            target[i_batch]=sample['label']
            image=cv2.imread(sample['filepath'])
            if self.au==False:
                image = cv2.resize(image, (self.size, self.size), cv2.INTER_CUBIC)
            else:
                rank=np.random.randint(0,100)/100
                if rank>0.5:
                    size1=np.random.randint(self.size+1,1+int(self.size*1.1))
                    image=randomCrop(image, crop_shape=(self.size,self.size),size=size1 )
                else:
                    image = cv2.resize(image, (self.size, self.size), cv2.INTER_CUBIC)
            image = image[:, :, ::-1]
            batch_inputs[i_batch] = preprocess_input(image)
            #if sample['label']==1:
            #    mask = cv2.imread(sample['maskpath'],0)
            #    mask = cv2.resize(mask, (self.size, self.size), cv2.INTER_CUBIC)
            #    batch_mask[i_batch] = mask.reshape((self.size,self.size,1))/255.
        return (batch_inputs ,target)
    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            self.data=self.data.sample(frac=1)
            self.data.index=[i for i in range(0,self.samples_num)]

class DataGeneratorM(keras.utils.Sequence):
#class DataGenerator:
    def __init__(self,data, batch_size=32,size=96, shuffle=True,au=False,mask_num=1):
        self.batch_size = batch_size
        self.size=size
        self.data=data
        self.shuffle = shuffle
        self.au=au
        self.mask_num=mask_num
        self.samples_num=self.data.shape[0]
        self.data.index = [i for i in range(0, self.samples_num)]

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(self.samples_num / float(self.batch_size))

    def __getitem__(self, index):
        i=index*self.batch_size
        length=min(self.batch_size,(self.samples_num-i))
        batch_inputs=np.zeros((length,self.size,self.size,3),dtype=np.float32)
        batch_mask = np.zeros((length, self.size, self.size, 1), dtype=np.float32)
        #target=np.zeros((length),dtype=np.float32)
        for i_batch in range(0,length):
            sample=self.data.iloc[i+i_batch].to_dict()
            #target[i_batch]=sample['label']
            image=cv2.imread(sample['filepath'])
            if sample['label']==1:
                mask = cv2.imread(sample['maskpath'], 0)
                mask[mask >= 127] = 255
                mask[mask < 127] = 0
            else:
                mask=np.zeros((self.size, self.size), dtype=np.uint8)
            image = image[:, :, ::-1]
            if self.au==False:
                image = cv2.resize(image, (self.size, self.size), cv2.INTER_CUBIC)
                mask = cv2.resize(mask, (self.size, self.size), cv2.INTER_CUBIC)
            else:
                rank = np.random.randint(0, 100) / 100
                if rank > 0.3:
                    size1 = np.random.randint(self.size + 1, 1 + int(self.size * 1.1))
                    image,mask = randomCropMask(image,mask, crop_shape=(self.size, self.size), size=size1)
                else:
                    image = cv2.resize(image, (self.size, self.size), cv2.INTER_CUBIC)
                    mask = cv2.resize(mask, (self.size, self.size), cv2.INTER_CUBIC)
            batch_inputs[i_batch] = preprocess_input(image)
            batch_mask[i_batch] = mask.reshape((self.size, self.size, 1)) / 255.
        if self.mask_num==6:
            return (batch_inputs , [batch_mask,batch_mask,batch_mask,batch_mask,batch_mask,batch_mask])
        else:
            return (batch_inputs, batch_mask)


    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            self.data=self.data.sample(frac=1)
            self.data.index=[i for i in range(0,self.samples_num)]

def iou(y_true, y_pred, label=0):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    #y_pred = K.cast(K.greater(y_pred, 0.5), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1] #- 1
    #print(num_labels)
    # initialize a variable to store total IoU in
    mean_iou =0.0 # K.variable(0.0)

    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        mean_iou = mean_iou + iou(y_true, y_pred, label)

    # divide total IoU by number of labels to get mean IoU
    return mean_iou / num_labels


if __name__ == '__main__':
    if os.path.exists('data')==False:
        os.makedirs('data')
    data_base = 'F:/data/mars/cancer/'
    df,test=getData(data_base)
    df.to_csv('data/gastric.csv',index=False)
    test.to_csv('data/gastric_test.csv', index=False)

