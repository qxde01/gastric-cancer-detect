import os,math
import pandas as pd
import numpy as np
from tensorflow import keras
#import tensorflow.keras.backend as K
import cv2
#import Augmentor
#data_base='F:/data/mars/cancer/'

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
    df['filename'] = [x.split('/')[-1] for x in df.filepath.values]
    df=df.sample(frac=1)
    ns=int(df.shape[0]*0.8)
    df['type']='train'
    df['type'].values[ns:]='test'
    print(df.groupby(['type','label'])['filepath'].count())
    flist3 = os.listdir(data_base + 'test/')
    test = pd.DataFrame([data_base + 'test/' + x for x in flist3], columns=['filepath'])
    test['filename'] = flist3
    return df,test

def bandFilter(img,w=2,radius=100):
    #傅里叶变换
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)
    #设置带通滤波器
    # w 带宽
    # radius: 带中心到频率平面原点的距离
    rows, cols = img.shape
    crow,ccol = int(rows/2), int(cols/2) #中心位置
    #w = 25
    #radius =25
    mask = np.ones((rows, cols, 2), np.uint8)
    for i in range(0, rows):
        for j in range(0, cols):
            # 计算(i, j)到中心点的距离
            d = math.sqrt(pow(i - crow, 2) + pow(j - ccol, 2))
            if radius - w / 2 < d < radius + w / 2:
                mask[i, j, 0] = mask[i, j, 1] = 0
            else:
                mask[i, j, 0] = mask[i, j, 1] = 1
    #掩膜图像和频谱图像乘积
    f = fshift * mask
    #傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])
    res = 255 * (res - res.min()) / (res.max() - res.min())
    return res.astype(np.uint8)

def preprocess_input(x):
    #[:,:,0]=bandFilter(x[:,:,0],w=2,radius=100)
    #x[:, :, 1] = bandFilter(x[:, :, 1], w=2, radius=100)
    #x[:, :, 2] = bandFilter(x[:, :, 2], w=2, radius=100)
    #if x.dtype not in ['float32', 'float64', 'float']:
    x = x.astype(np.float32)
    x /= 127.5
    x -= 1.
    #x /=255.
    #x=0.01+0.98*x
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
                if rank > 0.85:
                    size1 = np.random.randint(self.size + 1, 1 + int(self.size * 1.1))
                    image,mask = randomCropMask(image,mask, crop_shape=(self.size, self.size), size=size1)
                elif rank >0.75 and rank <0.85:
                    image = cv2.resize(image, (self.size, self.size), cv2.INTER_CUBIC)
                    mask = cv2.resize(mask, (self.size, self.size), cv2.INTER_CUBIC)
                    image = np.fliplr(image)
                    mask=np.fliplr(mask)
                elif rank >0.65 and rank <0.75:
                    image = cv2.resize(image, (self.size, self.size), cv2.INTER_CUBIC)
                    mask = cv2.resize(mask, (self.size, self.size), cv2.INTER_CUBIC)
                    image = np.flipud(image)
                    mask=np.flipud(mask)
                elif rank > 0.2 and rank <0.65:
                    image = cv2.resize(image, (self.size, self.size), cv2.INTER_CUBIC)
                    mask = cv2.resize(mask, (self.size, self.size), cv2.INTER_CUBIC)
                    center = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), np.random.randint(10, 180, 1), 1)
                    image = cv2.warpAffine(image, center, (self.size, self.size))
                    mask = cv2.warpAffine(mask, center, (self.size, self.size))
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



if __name__ == '__main__':
    if os.path.exists('data')==False:
        os.makedirs('data')
    data_base = 'F:/data/mars/cancer/'
    df,test=getData(data_base)
    df.to_csv('data/gastric.csv',index=False)
    test.to_csv('data/gastric_test.csv', index=False)

