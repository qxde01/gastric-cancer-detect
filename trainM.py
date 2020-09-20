from tensorflow import keras
import pandas as pd
import sys,argparse
from models import build_model
from data import DataGeneratorM
from  utils import IOU,DICE
from  models.losses import getLoss
from sch import WarmUpLearningRateScheduler,CosineAnnealingScheduler
import tensorflow_addons as tfa
#from bert4keras.optimizers import Adam,extend_with_weight_decay,extend_with_gradient_accumulation


def modelSave(net,size=256,name=None,num_mask=1):
    '''
    if name==None:
        if num_mask==1:
            filepath = 'saved/%s_%s_{val_loss:.4f}-{val_DICE:.4f}-{epoch:03d}.h5' % (net, size)
        else:
            filepath = 'saved/%s_%s_{val_loss:.4f}-{val_d_DICE:.4f}-{epoch:03d}.h5' % (net, size)
    else:
    '''
    if num_mask==1:
            filepath = 'saved/%s_%s_%s_{val_loss:.4f}-{val_DICE:.4f}-{epoch:03d}.h5' % (net, size,name)
    else:
            filepath = 'saved/%s_%s_%s_{val_loss:.4f}-{val_d_DICE:.4f}-{epoch:03d}.h5' % (net, size,name)

    return keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, save_best_only=True)


def get_args():
    parser = argparse.ArgumentParser(description="net .", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--net", "-net",type=str,  help=" net type",default='U2netS')
    parser.add_argument("--batch_size","-batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--epochs","-epochs", type=int, default=150, help="number of epochs")
    parser.add_argument("--size", "-size", type=int, default=256, help=" image size")
    parser.add_argument('--pretrained',"-p",type=str,default=None,help=' pretrained model weights  h5 file')
    parser.add_argument("--learning_rate",'-lr', type=float,default=0.001, help="learning_rate")
    parser.add_argument("--input", '-input', type=str, default='data/gastric.csv', help="train data csv")
    parser.add_argument("--warmup", '-wp', type=int, default=1, help="warmup ")
    parser.add_argument("--loss",'-loss', type=str,default='bce_dice_focal',help="loss")
    parser.add_argument("--optimizers", '-opt', type=str, default='SGD', help="optimizers")
    parser.add_argument("--lrs", '-lrs', type=int, default=0, help="lr decay,1-cos ")
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
    d0 = d[d.label == 0].sample(frac=0.2)
    d0.type='train'
    d=pd.concat([d0,d1]).sample(frac=1)
    nn=d[d.type=='train'].shape[0]
    test=d[d.type=='test']
    test=test[test.label==1]
    net =parsers.net
    opt=parsers.optimizers
    wp=parsers.warmup
    lrs=parsers.lrs
    model = build_model(size=size, net=net)

    mask_num=len(model.outputs)

    #if net in ['Unet','Deeplabv3']:
    #    mask_num=1
    #else:
    #    mask_num=6
    train_gen=DataGeneratorM(data=d[d.type=='train'],batch_size=batch_size,size=size,au=True,mask_num=mask_num)
    val_gen=DataGeneratorM(data=test,batch_size=batch_size,size=size,shuffle=False,mask_num=mask_num)

    model=build_model(size=size, net=net)
    if parsers.pretrained is not None:
        model.load_weights(parsers.pretrained)
    #model.summary()

    if opt=='SGD':
        opt=keras.optimizers.SGD(learning_rate=lr,momentum=0.95,nesterov=True)
    elif opt=='Adam':
        opt=keras.optimizers.Adam(learning_rate=lr,amsgrad=True)
    elif opt=='SGDW':
        opt =tfa.optimizers.SGDW(learning_rate=lr,weight_decay=0.00005,momentum=0.95,nesterov=True)
    elif opt=='AdamW':
        opt=tfa.optimizers.AdamW(learning_rate=lr,weight_decay=0.00005,amsgrad=True)
    #elif opt=='AdamWG':
    #    AdamW = extend_with_weight_decay(Adam, 'AdamW')
    #    AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
    #    opt = AdamWG(learning_rate=lr,weight_decay_rate=0.0,exclude_from_weight_decay=['Norm', 'bias'],grad_accum_steps=4 )
    else:
        opt=keras.optimizers.SGD(learning_rate=lr)

    print(parsers)
    name=parsers.loss
    model.compile(optimizer=opt,loss=getLoss(name=name),metrics=[DICE] )
    #model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy,metrics=[DICE])
    callbacks_list = [modelSave(net, size=size, name=name,num_mask=mask_num),keras.callbacks.EarlyStopping(patience=50)]
    if wp>0:
        callbacks_list=callbacks_list+[WarmUpLearningRateScheduler(warmup_batches=wp*nn//batch_size,init_lr=0.00001)]
    if lrs>0:
        callbacks_list=callbacks_list+[CosineAnnealingScheduler(T_max=epochs, eta_max=lr, eta_min=0.000001, verbose=1)]

    model.fit(train_gen,epochs=epochs,steps_per_epoch=nn//batch_size,validation_data=val_gen,callbacks=callbacks_list,workers=3 )

