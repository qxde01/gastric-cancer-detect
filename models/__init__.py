import sys
from .Unet import Unet
from .Unet3 import Unet3
from .U2net import U2net,U2netS,U2netM,U2netSP
from .UEfficientNet import UEfficientNetB4
from .UMFacenet2 import UMFacenet
from .SqueezeUNet import SqueezeUNet
from .mobilenet_v3 import MobileNetV3Small
from .deeplab_v3 import Deeplabv3

def build_model(size=256,net='U2netS'):
    if net=='U2netS':
        model = U2netS(input_shape=(size, size, 3), drop_rate=0.5)
    elif net=='U2netM':
        model = U2netM(input_shape=(size, size, 3))
    elif net=='U2netSP':
        model = U2netSP(input_shape=(size, size, 3), drop_rate=0.5)
    elif net=='U2net':
        model = U2net(input_shape=(size, size, 3))
    elif net=='Unet3':
        model = Unet3(input_shape=(size, size, 3))
    elif net=='UMFacenet':
        model = UMFacenet(input_shape=(size, size, 3))
    elif net == 'UMFacenetS':
        model = UMFacenet(input_shape=(size, size, 3), filters=[32,64,128,256],  use_se=False)
    elif net=='Unet':
        model = Unet(input_shape=(size, size, 3))
    elif net=='UEfficientNetB4':
        model = UEfficientNetB4(input_shape=(size, size, 3),imagenet_weights='imagenet')
    elif net=='SqueezeUNet':
        model=SqueezeUNet(input_shape=(size, size, 3))
    elif net =='MobileNetV3Small':
        model=MobileNetV3Small(input_shape=(size, size, 3))
    elif net=='Deeplabv3':
        model = Deeplabv3(input_shape=(size, size, 3), classes=1,backbone='mobilenetv2', OS=16, alpha=1., activation='sigmoid')
    else:
        print(' not support your net .')
        sys.exit()
    return model
