from tensorflow import keras
#https://github.com/NathanUA/U-2-Net
#https://arxiv.org/pdf/2005.09007v1.pdf
def conv_block(inputs, filters, kernel_size=3, strides=1,dirate=1, padding='same'):
    Z = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False,dilation_rate=(dirate,dirate))(inputs)
    Z = keras.layers.BatchNormalization(axis=-1)(Z)
    #A = keras.layers.PReLU(shared_axes=[1, 2])(Z)
    Z=keras.layers.ReLU()(Z)
    return Z

def RSU7(inputs,out_filters=3,mid_filters=12):
    hxin=conv_block(inputs,out_filters)
    hx1=conv_block(hxin,mid_filters)
    hx=keras.layers.MaxPool2D(pool_size=(2,2))(hx1)

    hx2 = conv_block(hx, mid_filters)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx2)

    hx3 = conv_block(hx, mid_filters)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx3)

    hx4 = conv_block(hx, mid_filters)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx4)

    hx5 = conv_block(hx, mid_filters)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx5)

    hx6 = conv_block(hx, mid_filters)
    hx7 = conv_block(hx6, mid_filters)

    hx6d=keras.layers.concatenate([hx6,hx7])
    hx6d=conv_block(hx6d, mid_filters)
    hx6dup=keras.layers.UpSampling2D(size=(2,2),interpolation='bilinear')(hx6d)

    hx5d = keras.layers.concatenate([hx6dup, hx5])
    hx5d = conv_block(hx5d, mid_filters )
    hx5dup = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx5d)

    hx4d = keras.layers.concatenate([hx5dup, hx4])
    hx4d = conv_block(hx4d, mid_filters)
    hx4dup = keras.layers.UpSampling2D(size=(2, 2))(hx4d)

    hx3d = keras.layers.concatenate([hx4dup, hx3])
    hx3d = conv_block(hx3d, mid_filters)
    hx3dup = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx3d)

    hx2d = keras.layers.concatenate([hx3dup, hx2])
    hx2d = conv_block(hx2d, mid_filters )
    hx2dup = keras.layers.UpSampling2D(size=(2, 2))(hx2d)

    hx1d = keras.layers.concatenate([hx2dup, hx1])
    hx1d = conv_block(hx1d, out_filters )
    hxou=keras.layers.add([hxin,hx1d])
    return hxou

def RSU6(inputs,out_filters=3,mid_filters=12):
    hxin=conv_block(inputs,out_filters)
    hx1=conv_block(hxin,mid_filters)
    hx=keras.layers.MaxPool2D(pool_size=(2,2))(hx1)

    hx2 = conv_block(hx, mid_filters)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx2)

    hx3 = conv_block(hx, mid_filters)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx3)

    hx4 = conv_block(hx, mid_filters)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx4)

    hx5 = conv_block(hx, mid_filters)
    hx6 = conv_block(hx5, mid_filters,dirate=2)

    hx5d = keras.layers.concatenate([hx6, hx5])
    hx5d = conv_block(hx5d, mid_filters)
    hx5dup = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx5d)

    hx4d = keras.layers.concatenate([hx5dup, hx4])
    hx4d = conv_block(hx4d, mid_filters)
    hx4dup = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx4d)

    hx3d = keras.layers.concatenate([hx4dup, hx3])
    hx3d = conv_block(hx3d, mid_filters)
    hx3dup = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx3d)

    hx2d = keras.layers.concatenate([hx3dup, hx2])
    hx2d = conv_block(hx2d, mid_filters)
    hx2dup = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx2d)

    hx1d = keras.layers.concatenate([hx2dup, hx1])
    hx1d = conv_block(hx1d, out_filters)
    hxou = keras.layers.add([hxin, hx1d])
    return hxou

def RSU5(inputs,out_filters=3,mid_filters=12):
    hxin=conv_block(inputs,out_filters)
    hx1=conv_block(hxin,mid_filters)
    hx=keras.layers.MaxPool2D(pool_size=(2,2))(hx1)

    hx2 = conv_block(hx, mid_filters)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx2)

    hx3 = conv_block(hx, mid_filters)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx3)

    hx4 = conv_block(hx, mid_filters)
    hx5 = conv_block(hx4, mid_filters,dirate=2)

    hx4d = keras.layers.concatenate([hx5, hx4])
    hx4d = conv_block(hx4d, mid_filters)
    hx4dup = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx4d)

    hx3d = keras.layers.concatenate([hx4dup, hx3])
    hx3d = conv_block(hx3d, mid_filters)
    hx3dup = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx3d)

    hx2d = keras.layers.concatenate([hx3dup, hx2])
    hx2d = conv_block(hx2d, mid_filters)
    hx2dup = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx2d)

    hx1d = keras.layers.concatenate([hx2dup, hx1])
    hx1d = conv_block(hx1d, out_filters)
    hxou = keras.layers.add([hxin, hx1d])
    return hxou

def RSU4(inputs,out_filters=3,mid_filters=12):
    hxin=conv_block(inputs,out_filters)
    hx1=conv_block(hxin,mid_filters)
    hx=keras.layers.MaxPool2D(pool_size=(2,2))(hx1)

    hx2 = conv_block(hx, mid_filters)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx2)

    hx3 = conv_block(hx, mid_filters)
    hx4 = conv_block(hx3, mid_filters,dirate=2)

    hx3d = keras.layers.concatenate([hx4, hx3])
    hx3d = conv_block(hx3d, mid_filters)
    hx3dup = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx3d)

    hx2d = keras.layers.concatenate([hx3dup, hx2])
    hx2d = conv_block(hx2d, mid_filters)
    hx2dup = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx2d)

    hx1d = keras.layers.concatenate([hx2dup, hx1])
    hx1d = conv_block(hx1d, out_filters)
    hxou = keras.layers.add([hxin, hx1d])
    return hxou

def RSU4F(inputs,out_filters=3,mid_filters=12):
    hxin=conv_block(inputs,out_filters) #shape=(None, 20, 20, 256)
    hx1=conv_block(hxin,mid_filters) #shape=(None, 20, 20, 512)
    hx2 = conv_block(hx1, mid_filters,dirate=2) #shape=(None, 20, 20, 512)
    hx3 = conv_block(hx2, mid_filters,dirate=4) #shape=(None, 20, 20, 512)
    hx4 = conv_block(hx3, mid_filters,dirate=8) #shape=(None, 20, 20, 512)

    hx3d = keras.layers.concatenate([hx4, hx3]) #shape=(None, 20, 20, 1024)
    hx3d = conv_block(hx3d, mid_filters,dirate=4) #shape=(None, 20, 20, 512)

    hx2d = keras.layers.concatenate([hx3d, hx2]) #shape=(None, 20, 20, 1024)
    hx2d = conv_block(hx2d, mid_filters, dirate=2) #shape=(None, 20, 20, 512)

    hx1d = keras.layers.concatenate([hx2d, hx1]) # shape=(None, 20, 20, 1024
    hx1d = conv_block(hx1d, mid_filters, dirate=1) #shape=(None, 20, 20, 512)

    hxou = keras.layers.add([hxin, hx1d])
    return hxou

def U2net(input_shape=(320,320,3)):
    inputs = keras.layers.Input(shape=input_shape)

    hx1=RSU7(inputs, 32, 64) #shape=(None, 320, 320, 32)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx1) #shape=(None, 160, 160, 32)

    hx2=RSU6(hx, 32, 128) #shape=(None, 160, 160, 32)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx2) #shape=(None, 80, 80, 32)

    hx3=RSU5(hx, 64, 256) #shape=(None, 80, 80, 64)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx3) #shape=(None, 40, 40, 64)

    hx4=RSU4(hx, 128, 512) #shape=(None, 40, 40, 128)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx4) #shape=(None, 20, 20, 128)

    # ?
    hx5=RSU4F(hx, 256, 256) #shape=(None, 20, 20, 256)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx5) #shape=(None, 10, 10, 256

    hx6 = RSU4F(hx, 512, 512) #shape=(None, 10, 10, 512)

    hx6up = keras.layers.UpSampling2D(size=(2, 2))(hx6) #shape=(None, 20, 20, 512)

    hx5d = keras.layers.concatenate([hx6up, hx5]) #shape=(None, 20, 20, 768)
    hx5d=RSU4F(hx5d,512,512) #shape=(None, 20, 20, 512)
    hx5up = keras.layers.UpSampling2D(size=(2, 2))(hx5d) #shape=(None, 40, 40, 512)

    hx4d = keras.layers.concatenate([hx5up, hx4]) # shape=(None, 40, 40, 640)
    hx4d = RSU4(hx4d, 128, 256) #shape=(None, 40, 40, 128)
    hx4up = keras.layers.UpSampling2D(size=(2, 2))(hx4d) #shape=(None, 80, 80, 128)

    hx3d = keras.layers.concatenate([hx4up, hx3]) #shape=(None, 80, 80, 192)
    hx3d = RSU5(hx3d, 64, 128) #shape=(None, 80, 80, 64)
    hx3up = keras.layers.UpSampling2D(size=(2, 2))(hx3d) #shape=(None, 160, 160, 64)

    hx2d = keras.layers.concatenate([hx3up, hx2]) #shape=(None, 160, 160, 96)
    hx2d = RSU6(hx2d, 32, 64) #shape=(None, 160, 160, 32)
    hx2up = keras.layers.UpSampling2D(size=(2, 2))(hx2d) #shape=(None, 320, 320, 32)

    hx1d = keras.layers.concatenate([hx2up, hx1]) #shape=(None, 320, 320, 64)
    hx1d = RSU7(hx1d, 16, 64)

    d1 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hx1d)
    d11 = keras.layers.Activation('sigmoid', name='d1')(d1)

    d2 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hx2d)
    d2 = keras.layers.UpSampling2D(size=(2, 2))(d2)
    d22 = keras.layers.Activation('sigmoid', name='d2')(d2)

    d3 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hx3d) #shape=(None, 80, 80, 1)
    d3 = keras.layers.UpSampling2D(size=(4, 4))(d3)
    d33 = keras.layers.Activation('sigmoid', name='d3')(d3)

    d4 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)( hx4d)  # shape=(None, 40, 40, 1)
    d4 = keras.layers.UpSampling2D(size=(8, 8))(d4)
    d44 = keras.layers.Activation('sigmoid', name='d4')(d4)

    d5 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)( hx5d)  # shape=(None, 20, 20, 1)
    d5 = keras.layers.UpSampling2D(size=(16, 16))(d5)
    d55 = keras.layers.Activation('sigmoid', name='d5')(d5)

    d=keras.layers.concatenate([d1,d2,d3,d4,d5])
    d = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d)
    d = keras.layers.Activation('sigmoid', name='d')(d)
    '''
    Total params: 94,339,965
    Trainable params: 94,295,229
    Non-trainable params: 44,736
    '''
    model=keras.models.Model(inputs=inputs,outputs=[d,d11,d22,d33,d44,d55])
    return model

def U2netS(input_shape=(320,320,3),drop_rate=0.35):
    inputs = keras.layers.Input(shape=input_shape)
    hx1 = RSU7(inputs, 16, 64)  # shape=(None, 320, 320, 16)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx1)
    if drop_rate>0.:
        hx=keras.layers.Dropout(drop_rate)(hx)
    hx2 = RSU6(hx, 16, 64)  # shape=(None, 160, 160, 16)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx2)
    if drop_rate>0.:
        hx=keras.layers.Dropout(drop_rate)(hx)
    hx3 = RSU5(hx, 16, 64)  # shape=(None, 80, 80, 16)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx3)  # shape=(None, 40, 40, 16)
    if drop_rate>0.:
        hx=keras.layers.Dropout(drop_rate)(hx)
    hx4 = RSU4(hx, 16, 64)  # shape=(None, 40, 40, 16)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx4)

    hx5 = RSU4F(hx, 16, 16)  # shape=(None, 20, 20, 16)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx5)  # shape=(None, 10, 10, 16
    if drop_rate>0.:
        hx=keras.layers.Dropout(drop_rate)(hx)
    hx6 = RSU4F(hx, 16, 16)  # shape=(None, 10, 10, 16)
    hx6up = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx6)

    hx5d = keras.layers.concatenate([hx6up, hx5])
    hx5d = RSU4F(hx5d, 16, 16)
    hx5up = keras.layers.UpSampling2D(size=(2, 2))(hx5d)

    hx4d = keras.layers.concatenate([hx5up, hx4])
    #if drop_rate>0.:
    #    hx4d=keras.layers.Dropout(drop_rate)(hx4d)
    hx4d = RSU4(hx4d, 16, 64)
    hx4up = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx4d)

    hx3d = keras.layers.concatenate([hx4up, hx3])
    hx3d = RSU5(hx3d, 16, 64)
    hx3up = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx3d)

    hx2d = keras.layers.concatenate([hx3up, hx2])
    #if drop_rate>0.:
    #    hx2d=keras.layers.Dropout(drop_rate)(hx2d)
    hx2d = RSU6(hx2d, 16, 64)
    hx2up = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx2d)

    hx1d = keras.layers.concatenate([hx2up, hx1])
    #if drop_rate>0.:
    #    hx1d=keras.layers.Dropout(drop_rate)(hx1d)

    hx1d = RSU7(hx1d, 16, 64)

    d1 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hx1d)
    d11 = keras.layers.Activation('sigmoid', name='d1')(d1)

    d2 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hx2d)
    d2 = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(d2)
    d22 = keras.layers.Activation('sigmoid', name='d2')(d2)

    d3 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(
        hx3d)  # shape=(None, 80, 80, 1)
    d3 = keras.layers.UpSampling2D(size=(4, 4),interpolation='bilinear')(d3)
    d33 = keras.layers.Activation('sigmoid', name='d3')(d3)

    d4 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(
        hx4d)  # shape=(None, 40, 40, 1)
    d4 = keras.layers.UpSampling2D(size=(8, 8))(d4)
    d44 = keras.layers.Activation('sigmoid', name='d4')(d4)

    d5 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(
        hx5d)  # shape=(None, 20, 20, 1)
    d5 = keras.layers.UpSampling2D(size=(16, 16),interpolation='bilinear')(d5)
    d55 = keras.layers.Activation('sigmoid', name='d5')(d5)

    d = keras.layers.concatenate([d1, d2, d3, d4, d5])
    d = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d)
    d = keras.layers.Activation('sigmoid', name='d')(d)
    '''
   Total params: 3,738,541
   Trainable params: 3,728,045
   Non-trainable params: 10,496
    '''
    model = keras.models.Model(inputs=inputs, outputs=[d, d11, d22, d33, d44, d55])
    return model



def U2netM(input_shape=(320,320,3)):
    inputs = keras.layers.Input(shape=input_shape)

    hx1=RSU7(inputs, 32, 64) #shape=(None, 320, 320, 32)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx1) #shape=(None, 160, 160, 32)

    hx2=RSU6(hx, 32, 64) #shape=(None, 160, 160, 32)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx2) #shape=(None, 80, 80, 32)

    hx3=RSU5(hx, 32, 128) #shape=(None, 80, 80, 32)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx3) #shape=(None, 40, 40, 64)

    hx4=RSU4(hx, 64, 128) #shape=(None, 40, 40, 64)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx4) #shape=(None, 20, 20, 128)

    # ?
    hx5=RSU4F(hx, 128, 128) #shape=(None, 20, 20, 128)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx5) #shape=(None, 10, 10, 256

    hx6 = RSU4F(hx, 256, 256) #shape=(None, 10, 10, 256)

    hx6up = keras.layers.UpSampling2D(size=(2, 2))(hx6) #shape=(None, 20, 20, 256)

    hx5d = keras.layers.concatenate([hx6up, hx5]) #shape=(None, 20, 20, 384)
    hx5d=RSU4F(hx5d,256,256) #shape=(None, 20, 20, 256)
    hx5up = keras.layers.UpSampling2D(size=(2, 2))(hx5d) #shape=(None, 40, 40, 512)

    hx4d = keras.layers.concatenate([hx5up, hx4]) # shape=(None, 40, 40, 640)
    hx4d = RSU4(hx4d, 128, 256) #shape=(None, 40, 40, 128)
    hx4up = keras.layers.UpSampling2D(size=(2, 2))(hx4d) #shape=(None, 80, 80, 128)

    hx3d = keras.layers.concatenate([hx4up, hx3]) #shape=(None, 80, 80, 192)
    hx3d = RSU5(hx3d, 32, 128) #shape=(None, 80, 80, 32)
    hx3up = keras.layers.UpSampling2D(size=(2, 2))(hx3d) #shape=(None, 160, 160, 64)

    hx2d = keras.layers.concatenate([hx3up, hx2]) #shape=(None, 160, 160, 96)
    hx2d = RSU6(hx2d, 32, 64) #shape=(None, 160, 160, 32)
    hx2up = keras.layers.UpSampling2D(size=(2, 2))(hx2d) #shape=(None, 320, 320, 32)

    hx1d = keras.layers.concatenate([hx2up, hx1]) #shape=(None, 320, 320, 64)
    hx1d = RSU7(hx1d, 16, 64)

    d1 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hx1d)
    d11 = keras.layers.Activation('sigmoid', name='d1')(d1)

    d2 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hx2d)
    d2 = keras.layers.UpSampling2D(size=(2, 2))(d2)
    d22 = keras.layers.Activation('sigmoid', name='d2')(d2)

    d3 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hx3d) #shape=(None, 80, 80, 1)
    d3 = keras.layers.UpSampling2D(size=(4, 4))(d3)
    d33 = keras.layers.Activation('sigmoid', name='d3')(d3)

    d4 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)( hx4d)  # shape=(None, 40, 40, 1)
    d4 = keras.layers.UpSampling2D(size=(8, 8))(d4)
    d44 = keras.layers.Activation('sigmoid', name='d4')(d4)

    d5 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)( hx5d)  # shape=(None, 20, 20, 1)
    d5 = keras.layers.UpSampling2D(size=(16, 16))(d5)
    d55 = keras.layers.Activation('sigmoid', name='d5')(d5)

    d=keras.layers.concatenate([d1,d2,d3,d4,d5])
    d = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d)
    d = keras.layers.Activation('sigmoid', name='d')(d)
    '''
    Total params: 26,829,661
    Trainable params: 26,803,613
    Non-trainable params: 26,048

    '''
    model=keras.models.Model(inputs=inputs,outputs=[d,d11,d22,d33,d44,d55])
    return model


def U2netSP(input_shape=(320,320,3),drop_rate=0.35):
    inputs = keras.layers.Input(shape=input_shape)
    hx1 = RSU7(inputs, 32, 64)  # shape=(None, 320, 320, 16)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx1)
    if drop_rate>0.:
        hx=keras.layers.Dropout(drop_rate)(hx)
    hx2 = RSU6(hx, 32, 64)  # shape=(None, 160, 160, 16)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx2)
    if drop_rate>0.:
        hx=keras.layers.Dropout(drop_rate)(hx)
    hx3 = RSU5(hx, 32, 64)  # shape=(None, 80, 80, 16)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx3)  # shape=(None, 40, 40, 16)
    if drop_rate>0.:
        hx=keras.layers.Dropout(drop_rate)(hx)
    hx4 = RSU4(hx, 32, 64)  # shape=(None, 40, 40, 16)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx4)

    hx5 = RSU4F(hx, 32, 32)  # shape=(None, 20, 20, 16)
    hx = keras.layers.MaxPool2D(pool_size=(2, 2))(hx5)  # shape=(None, 10, 10, 16
    if drop_rate>0.:
        hx=keras.layers.Dropout(drop_rate)(hx)
    hx6 = RSU4F(hx, 32, 32)  # shape=(None, 10, 10, 16)
    hx6up = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx6)

    hx5d = keras.layers.concatenate([hx6up, hx5])
    hx5d = RSU4F(hx5d, 32, 32)
    hx5up = keras.layers.UpSampling2D(size=(2, 2))(hx5d)

    hx4d = keras.layers.concatenate([hx5up, hx4])
    #if drop_rate>0.:
    #    hx4d=keras.layers.Dropout(drop_rate)(hx4d)
    hx4d = RSU4(hx4d, 32, 64)
    hx4up = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx4d)

    hx3d = keras.layers.concatenate([hx4up, hx3])
    hx3d = RSU5(hx3d, 32, 64)
    hx3up = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx3d)

    hx2d = keras.layers.concatenate([hx3up, hx2])
    #if drop_rate>0.:
    #    hx2d=keras.layers.Dropout(drop_rate)(hx2d)
    hx2d = RSU6(hx2d, 32, 64)
    hx2up = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(hx2d)

    hx1d = keras.layers.concatenate([hx2up, hx1])
    #if drop_rate>0.:
    #    hx1d=keras.layers.Dropout(drop_rate)(hx1d)

    hx1d = RSU7(hx1d, 32, 64)

    d1 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hx1d)
    d11 = keras.layers.Activation('sigmoid', name='d1')(d1)

    d2 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(hx2d)
    d2 = keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(d2)
    d22 = keras.layers.Activation('sigmoid', name='d2')(d2)

    d3 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(
        hx3d)  # shape=(None, 80, 80, 1)
    d3 = keras.layers.UpSampling2D(size=(4, 4),interpolation='bilinear')(d3)
    d33 = keras.layers.Activation('sigmoid', name='d3')(d3)

    d4 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(
        hx4d)  # shape=(None, 40, 40, 1)
    d4 = keras.layers.UpSampling2D(size=(8, 8))(d4)
    d44 = keras.layers.Activation('sigmoid', name='d4')(d4)

    d5 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(
        hx5d)  # shape=(None, 20, 20, 1)
    d5 = keras.layers.UpSampling2D(size=(16, 16),interpolation='bilinear')(d5)
    d55 = keras.layers.Activation('sigmoid', name='d5')(d5)

    d = keras.layers.concatenate([d1, d2, d3, d4, d5,inputs])
    d = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d)
    d = keras.layers.Activation('sigmoid', name='d')(d)
    '''
     Total params: 4,274,504
     Trainable params: 4,262,728
     Non-trainable params: 11,776
    '''
    model = keras.models.Model(inputs=inputs, outputs=[d, d11, d22, d33, d44, d55])
    return model


if __name__ == "__main__":
    import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model=U2netM(input_shape=(320,320,3))
    model.summary()
    keras.utils.plot_model(model, 'U2netM.png', show_shapes=True)
