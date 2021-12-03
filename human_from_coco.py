from pycocotools.coco import COCO
import zipfile
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2


def buffer2array(Z, image_name):
    '''
    无需解压，直接获取图片数据

    参数
    ===========
    Z:: 图片数据是 ZipFile 对象
    '''
    buffer = Z.read(image_name)
    image = np.frombuffer(buffer, dtype="B")  # 将 buffer 转换为 np.uint8 数组
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return img

def getMask(Z,imgid=262145,datatype='train2014'):
    img0 = coco.loadImgs(imgid)[0]
    img_name='%s/'%datatype +img0['file_name']
    img=buffer2array(Z, img_name)
    mask=np.zeros(img.shape[:2])
    annIds = coco.getAnnIds(imgIds=img0['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    for an in anns:
        mask0= coco.annToMask(an)
        mask[mask0 > 0] = 255
    return img,mask,img_name

def creat_dir(dir='a'):
    if os.path.exists(dir)==False:
        os.mkdir(dir)


if __name__ == '__main__':
    datatype='val2017'
    root_dir='F:/coco/'
    creat_dir(dir=root_dir+datatype)
    creat_dir(dir=root_dir + datatype+'_mask')
    Z = zipfile.ZipFile('%s%s.zip' %(root_dir,datatype) )
    annFile = '%sannotations/instances_%s.json' %(root_dir,datatype)
    coco = COCO(annFile)
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    for imgid in imgIds:
        img,mask,img_name=getMask(Z, imgid=imgid,datatype=datatype)
        imgpath=root_dir+img_name
        maskpath=root_dir + datatype+'_mask/'+img_name.split('/')[-1]
        print(imgpath,maskpath)
        cv2.imwrite(imgpath,img)
        cv2.imwrite(maskpath,mask)
        #cv2.imshow('img', img)
        #cv2.imshow('mask', mask)

    ''' 
    img_b = Z.read(Z.namelist()[1139])
    img=np.frombuffer(img_b, 'B')
    img_cv = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
    cv2.imshow('a',img_cv);cv2.waitKey(0)
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    imgId = coco.getImgIds(imgIds=[262145])
    img0 = coco.loadImgs(imgIds[ 0])[0]
    img_b = Z.read('train2014/'+img0['file_name'])
    plt.axis('off')
    plt.imshow(img_cv)
    plt.show()
    annIds = coco.getAnnIds(imgIds=img0['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    mask_single = coco.annToMask(anns[0])
    mask_single[mask_single>0]=255
    '''
