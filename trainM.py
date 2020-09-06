from tensorflow import keras
import pandas as pd
import sys,argparse
from models.Unet import Unet
from models.Unet3 import Unet3
from models.U2net import U2net,U2netS,U2netM
from models.UEfficientNet import UEfficientNetB4
from data import DataGeneratorM
from  utils import IOU,bce_dice_loss,DICE
import tensorflow_addons as tfa

def build_model(size=256,net='U2netS'):
    if net=='U2netS':
        model = U2netS(input_shape=(size, size, 3), drop_rate=0.0)
    elif net=='U2netM':
        model = U2netM(input_shape=(size, size, 3))
    elif net=='U2net':
        model = U2net(input_shape=(size, size, 3))
    elif net=='Unet3':
        model = Unet3(input_shape=(size, size, 3))
    elif net=='Unet':
        model = Unet(input_shape=(size, size, 3))
    elif net=='UEfficientNetB4':
        model = UEfficientNetB4(input_shape=(size, size, 3),imagenet_weights='saved/efficientnet-b4_imagenet_1000_notop.h5')
    else:
        print(' not support your net .')
        sys.exit()
    return model

def get_args():
    parser = argparse.ArgumentParser(description="net .", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--net", "-net",type=str,  help=" net type",default='U2netS')
    parser.add_argument("--batch_size","-batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--epochs","-epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--size", "-size", type=int, default=288, help=" image size")
    parser.add_argument('--pretrained',"-p",type=str,default=None,help=' pretrained model weights  h5 file')
    parser.add_argument("--learning_rate",'-lr', type=float,default=0.001, help="learning_rate")
    parser.add_argument("--input", '-input', type=str, default='data/gastric.csv', help="train data csv")
    #parser.add_argument("--warmup", '-wp', type=int, default=1, help="warmup ")
    #parser.add_argument("--lr_scheduler",'-lrs', type=int,default=2,help="1-stage decay,2-CosineAnnealingScheduler")
    parser.add_argument("--optimizers", '-opt', type=str, default='SGD', help="optimizers")
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    parsers=get_args()

    size=parsers.size
    batch_size=parsers.batch_size
    epochs=parsers.epochs
    lr=parsers.learning_rate
    d=pd.read_csv(parsers.input)
    d1=d[d.label==1]
    d0 = d[d.label == 0].sample(frac=0.35)
    d0.type='train'
    d=pd.concat([d0,d1]).sample(frac=1)
    nn=d[d.type=='train'].shape[0]
    net =parsers.net
    opt=parsers.optimizers
    if net in ['U2netS','U2netM','U2net','Unet3','UEfficientNetB4']:
        mask_num=6
    else:
        mask_num=1
    train_gen=DataGeneratorM(data=d[d.type=='train'],batch_size=batch_size,size=size,au=True,mask_num=mask_num)
    val_gen=DataGeneratorM(data=d[d.type=='test'],batch_size=batch_size,size=size,shuffle=False,mask_num=mask_num)

    model=build_model(size=size, net=net)
    if parsers.pretrained is not None:
        model.load_weights(parsers.pretrained)
    model.summary()

    if opt=='SGD':
        opt=keras.optimizers.SGD(learning_rate=lr,momentum=0.95,nesterov=True)
    elif opt=='Adam':
        opt=keras.optimizers.Adam(learning_rate=lr)
    elif opt=='SGDW':
        opt =tfa.optimizers.SGDW(learning_rate=lr,weight_decay=0.00005)
    elif opt=='AdamW':
        opt=tfa.optimizers.AdamW(learning_rate=lr,weight_decay=0.00005)
    else:
        opt=keras.optimizers.SGD(learning_rate=lr)

    print(parsers)

    filepath='saved/%s_%s_{val_loss:.4f}-{val_d_DICE:.4f}-{epoch:03d}.h5' % (net,size)

    model.compile(optimizer=opt,loss=bce_dice_loss,metrics=[DICE] )

    callbacks_list=[keras.callbacks.ModelCheckpoint(filepath,save_weights_only=True,save_best_only=False)]
                    #keras.callbacks.TensorBoard(log_dir='./logs',histogram_freq=10)]
    model.fit(train_gen,epochs=epochs,steps_per_epoch=nn//batch_size,validation_data=val_gen,callbacks=callbacks_list,workers=2 )

