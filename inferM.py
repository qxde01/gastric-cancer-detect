import pandas as pd
import numpy as np
#from sklearn import metrics
import cv2,keras,time,os,gc,argparse
from data import DataGeneratorM
from utils import find_best_dice
from data import preprocess_input
from models import build_model

def img2arr(test,size=224):
    N=test.shape[0]
    XRGB = np.zeros((N, size, size, 1), dtype=np.float32)
    j=0
    for i in range(0,N):
        p = test.maskpath.values[i]
        mask=cv2.imread(p,0)
        mask = cv2.resize(mask, (size, size), cv2.INTER_CUBIC)
        XRGB[i, :, :, :] = mask.reshape(size,size,1)/255.
    return XRGB

def fillHole(im_in):
    im_floodfill = im_in.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    #im_out = im_in | im_floodfill_inv
    return im_floodfill_inv

def get_args():
    parser = argparse.ArgumentParser(description="net .", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--net", "-net",type=str,  help=" net type",default='U2netS')
    #parser.add_argument("--batch_size","-batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--output","-o", type=int, default=0, help="0-只评估dice，1-输出test结果")
    parser.add_argument("--size", "-size", type=int, default=288, help=" image size")
    parser.add_argument('--pretrained',"-p",type=str,default=None,help=' pretrained model weights  h5 file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    if os.path.exists('result')==False:
        os.makedirs('result')
    parsers = get_args()
    size = parsers.size
    net = parsers.net
    o=parsers.output
    d=pd.read_csv('data/gastric.csv')
    #size=288
    #train_gen=DataGeneratorC(data=d[d.type=='train'],batch_size=batch_size,size=size,au=True)
    val=d[d.type=='test']
    val=val[val.label==1]
    masks=img2arr(val,size=size)

    #val_gen=DataGeneratorM(data=val,batch_size=2,size=size,shuffle=False)
    model = build_model(size=size, net=net)
    mask_num = len(model.outputs)
    if parsers.pretrained is not None:
        model.load_weights(parsers.pretrained)
    val_gen = DataGeneratorM(data=val, batch_size=2, size=size, shuffle=False, mask_num=mask_num)
    if mask_num==1:
        #pred = model.predict(val_gen, verbose=1)
        infer_model=model
    else:
        infer_model = keras.models.Model(model.inputs, model.outputs[0])
        infer_model.summary()
    pred = infer_model.predict(val_gen, verbose=1)
    score,alpha=find_best_dice(masks, pred,step=0.001)
    del(masks,pred)
    test=pd.read_csv('data/gastric_test.csv')
    nn=test.shape[0]
    if o>0:
        sub = 'result/%s_%s_'%(net,size) + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        if os.path.exists(sub) == False:
            os.makedirs(sub)
        cost_time=[]
        for i in range(0,nn):
            path=test.filepath[i]
            fp = test.filename[i].split('.')[0]
            t0=time.time()
            image=cv2.imread(path)
            w, h = image.shape[:2]
            image = cv2.resize(image, (size, size), cv2.INTER_CUBIC)
            image = image[:, :, ::-1]
            image = preprocess_input(image).reshape(1, size, size, 3)
            pred=infer_model(image)
            pred=pred.numpy().reshape(size,size)
            print( '==> ',i,fp,pred.mean())
            pred[pred>=alpha]=1.
            pred[pred< alpha]=0
            pred=pred*255.
            pred=pred.astype(np.uint8)
            #mask=cv2.erode(pred,(7,7))
            mask=cv2.resize(pred,(h,w),cv2.INTER_CUBIC)
            mask[mask>0]=255
            cost_time.append(time.time()-t0)
            cv2.imwrite(sub+'/'+fp+'_mask.jpg',mask)
            del(mask,image,pred)
            gc.collect()
        print(' avg cost time :%.6f'%np.array(cost_time).mean() )




'''
sample=test[test.label==1].iloc[0].to_dict()
image=cv2.imread(sample['filepath'])
mask = cv2.imread(sample['maskpath'], 0)
w,h=mask.shape
image = cv2.resize(image, (size, size), cv2.INTER_CUBIC)
mask = cv2.resize(mask, (size, size), cv2.INTER_CUBIC)
mask2=cv2.resize(mask, (h, w), cv2.INTER_CUBIC)
dice_np(masks,pred,a=0.1)
image=preprocess_input(image).reshape(1,size,size,3)
pred=model.predict(image,verbose=1)
for i in range(0,6):
    pr=pred[i]
    tmp=pr.reshape(size,size)
    m=tmp.mean()
    tmp[tmp>=m]=1.
    print(i,m)
    tmp=255*tmp
    tmp=tmp.astype(np.uint8)
    cv2.imwrite('%02d.png' %i,tmp)
'''




