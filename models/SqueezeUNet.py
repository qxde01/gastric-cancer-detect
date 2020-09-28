from tensorflow import keras

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"
# Modular function for Fire Node
# https://github.com/rcmalli/keras-squeezenet/blob/master/keras_squeezenet/squeezenet.py
def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'
    channel_axis = 3
    x = keras.layers.Conv2D(squeeze, (1, 1), padding='valid', use_bias=True,kernel_initializer='he_normal',name=s_id + sq1x1)(x)
    x = keras.layers.BatchNormalization(axis=-1,momentum=0.95,epsilon=1e-5, name=s_id + sq1x1+'/bn')(x)
    #x = keras.layers.Activation('relu', name=s_id + relu + sq1x1)(x)
    x=keras.layers.LeakyReLU()(x)
    left = keras.layers.Conv2D(expand//2, (1, 1), padding='valid',  use_bias=True,kernel_initializer='he_normal',name=s_id + exp1x1)(x)
    left = keras.layers.BatchNormalization(axis=-1, momentum=0.95,epsilon=1e-5,name=s_id + exp1x1 + '/bn')(left)
    #left = keras.layers.Activation('relu', name=s_id + relu + exp1x1)(left)
    left = keras.layers.LeakyReLU()(left)
    right = keras.layers.Conv2D(expand//2, (3, 3), padding='same', use_bias=True,kernel_initializer='he_normal', name=s_id + exp3x3)(x)
    right = keras.layers.BatchNormalization(axis=-1,momentum=0.95,epsilon=1e-5,name=s_id + exp3x3 + '/bn')(right)
    #right = keras.layers.Activation('relu', name=s_id + relu + exp3x3)(right)
    right = keras.layers.LeakyReLU()(right)
    x = keras.layers.concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    #print(x)
    return x


def SqueezeUNet(input_shape=(256,256,3),dropout=0.3):
    """Instantiates the SqueezeNet architecture.
    """
    #filters=[96,128,256,384,512]
    #filters=[64,96,160,256,384]
    filters = [48, 64, 128, 192, 256]
    img_input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(filters[0], (3, 3), strides=(1, 1), padding='same',  use_bias=True,kernel_initializer='he_normal',name='conv1')(img_input) #shape=(None, 256, 256, 96)
    x=keras.layers.BatchNormalization(axis=-1,momentum=0.95,epsilon=1e-5,name='conv1_bn')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x) # shape=(None, 128, 128, 96)

    f1 = fire_module(x, fire_id=2, squeeze=16, expand=filters[1]) #shape=(None, 128, 128, 128)
    f2 = fire_module(f1, fire_id=3, squeeze=16, expand=filters[1]) #shape=(None, 128, 128, 128)
    f2= keras.layers.add([f2,f1],name='fire2_3') #shape=(None, 128, 128, 128)
    if dropout>0:
        f2=keras.layers.Dropout(dropout)(f2)
    f3= fire_module(f2, fire_id=4, squeeze=32, expand=filters[2])  # shape=(None, 128, 128, 256)
    f4 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3',padding='same')(f3) #shape=(None, 64, 64, 256)

    f5 = fire_module(f4, fire_id=5, squeeze=32, expand=filters[2])  # shape=(None, 64, 64, 256)
    f5 = keras.layers.add([f5, f4], name='fire4_5')
    if dropout>0:
        f5=keras.layers.Dropout(dropout)(f5)
    f6=fire_module(f5, fire_id=6, squeeze=48, expand=filters[3]) #shape=(None, 64, 64, 384)

    f7 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4', padding='same')(f6)


    f8 = fire_module(f7, fire_id=7, squeeze=48, expand=filters[3]) #shape=(None, 32, 32, 384) d
    f9 = keras.layers.add([f7,f8] ,name='fire5_6')   #
    if dropout>0.:
        f9=keras.layers.Dropout(dropout)(f9)
    f9 = fire_module(f9, fire_id=8, squeeze=64, expand=filters[4])  # shape=(None, 32, 32, 512

    f10 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool8')(f9) #shape=(None, 16, 16, 512)
    #x = fire_module(x, fire_id=8, squeeze=64, expand=256) #13, 13, 512
    f10 = fire_module(f10, fire_id=9, squeeze=64, expand=filters[4]) # shape=(None, 16, 16, 512)
    #if dropout>0.:
    #    f10=keras.layers.Dropout(dropout/2)(f10)

    up5=keras.layers.Conv2DTranspose(filters[4],kernel_size=1,strides=(2,2),padding='same',name='up5_conv2d_trans')(f10) #shape=(None, 32, 32, 512)
    up5d=keras.layers.Add()([up5,f9])
    up5d = keras.layers.LeakyReLU()(up5d)
    up5d = fire_module(up5d, fire_id=10, squeeze=48, expand=filters[3]) #shape=(None, 32, 32, 384)

    up4 = keras.layers.Conv2DTranspose(filters[3], kernel_size=1, strides=(2, 2), padding='same',name='up4_conv2d_trans')(up5d) #shape=(None, 64, 64, 384)
    up4d = keras.layers.Add()([up4, f6])
    up4d = keras.layers.LeakyReLU()(up4d)
    up4d = fire_module(up4d, fire_id=11, squeeze=32, expand=filters[2])

    up3 = keras.layers.Conv2DTranspose(filters[2], kernel_size=1, strides=(2, 2), padding='same',name='up3_conv2d_trans')(up4d) #shape=(None, 128, 128, 256)
    up3d = keras.layers.Add()([up3, f3])
    up3d = keras.layers.LeakyReLU()(up3d)


    up3d = fire_module(up3d, fire_id=12, squeeze=16, expand=filters[1])
    #up3d = fire_module(up3d, fire_id=13, squeeze=16, expand=128)

    up20 = keras.layers.Conv2DTranspose(filters[1], kernel_size=1, strides=(2, 2), padding='same',name='up2_conv2d_trans')(up3d)
    up2 = keras.layers.Conv2D(filters[0], (3, 3), strides=(1, 1), padding='same', use_bias=True, kernel_initializer='he_normal', name='conv_up2')(up20)
    up2 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv_up2_bn')(up2)
    up2 = keras.layers.LeakyReLU()(up2)

    d1 = keras.layers.Conv2D(1, (3, 3), padding="same", activation=None, use_bias=False)(up2)
    d11 = keras.layers.Activation('sigmoid', name='d1')(d1)

    d2 = keras.layers.Conv2D(1, (3, 3), padding="same", activation=None, use_bias=False)(up20)
    d22 = keras.layers.Activation('sigmoid', name='d2')(d2)

    d3 = keras.layers.Conv2DTranspose(filters[1], kernel_size=1, strides=(2, 2), padding='same',name='d3_conv2d_trans')(up3d)
    d3 = keras.layers.Conv2D(1, (3, 3), padding="same", activation=None, use_bias=False)(d3)
    d33 = keras.layers.Activation('sigmoid', name='d3')(d3)

    d4 = keras.layers.Conv2DTranspose(filters[2], kernel_size=1, strides=(4, 4), padding='same',name='d4_conv2d_trans')(up4d)
    #d4 = keras.layers.Conv2DTranspose(256, kernel_size=3, strides=(2, 2), padding='same')(d4)
    #d4 = keras.layers.UpSampling2D(size=(2, 2))(d4)
    d4 = keras.layers.Conv2D(1, (3, 3), padding="same", activation=None, use_bias=False)(d4)
    d44 = keras.layers.Activation('sigmoid', name='d4')(d4)

    #d5 = keras.layers.Conv2DTranspose(filters[3], kernel_size=1, strides=(8,8), padding='same',name='d5_conv2d_trans')(up5d)
    #d5 = keras.layers.Conv2DTranspose(384, kernel_size=3, strides=(2, 2), padding='same')(d5) #1327488
    #d5 = keras.layers.Conv2DTranspose(filters[3], kernel_size=1, strides=(2, 2), padding='same')(d5) #1327488
    #d5=keras.layers.UpSampling2D(size=(2,2))(d5)
    #d5 = keras.layers.Conv2D(1, (3, 3), padding="same", activation=None, use_bias=False)(d5)
    #d55 = keras.layers.Activation('sigmoid', name='d5')(d5)

    d = keras.layers.concatenate([d1, d2, d3, d4,  img_input])
    d = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d)
    d = keras.layers.Activation('sigmoid', name='d')(d)

    model = keras.models.Model(inputs=img_input, outputs=[d, d11, d22, d33, d44])

    return model


if __name__ == "__main__":
    #Total params: 1,767,464
    model=SqueezeUNet(input_shape=(256,256,3))
    model.summary()
    keras.utils.plot_model(model, 'SqueezeUNet.png', show_shapes=True)
