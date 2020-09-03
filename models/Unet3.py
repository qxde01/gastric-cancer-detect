from tensorflow import keras
# [pytorch-Unet3](https://github.com/ZJUGiveLab/UNet-Version),[paper](https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
def conv_block(inputs, filters, kernel_size=3, strides=1, padding='same'):
    Z = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(inputs)
    #Z = keras.layers.BatchNormalization(axis=-1)(Z)
    #A = keras.layers.PReLU(shared_axes=[1, 2])(Z)
    A=keras.layers.ReLU()(Z)
    return A

def UnetConv2(inputs,filters,n=2,kernel_size=3,stride=1,padding='same'):
    x=inputs
    for i in range(0,n+1):
        x=keras.layers.Conv2D(filters,kernel_size=kernel_size,strides=(stride,stride),padding=padding, use_bias=False)(x)
        #x=keras.layers.BatchNormalization()(x)
        #x=keras.layers.PReLU(shared_axes=[1, 2])(x)
        x=keras.layers.ReLU()(x)
    return x

def Unet3(input_shape=(320,320,3),n=2,filters = [32, 64, 128, 256, 512] ):
    #filters = [64, 128, 256, 512, 1024] #113,566,149
    #filters = [32, 64, 128, 256, 512] #31,079,653,n=1 23,868,805
    CatChannels = filters[0]
    CatBlocks = 5
    UpChannels = CatChannels * CatBlocks
    inputs=keras.layers.Input(shape=input_shape)
    h1 = UnetConv2(inputs, filters[0],n=n)

    h2 = keras.layers.MaxPool2D(strides=(2,2))(h1)
    h2 = UnetConv2(h2, filters[1],n=n) #shape=(None, 160, 160, 128)

    h3 = keras.layers.MaxPool2D(strides=(2, 2))(h2)
    h3 = UnetConv2(h3, filters[2],n=n) #shape=(None, 80, 80, 256)

    h4 = keras.layers.MaxPool2D(strides=(2, 2))(h3)
    h4 = UnetConv2(h4, filters[3],n=1) #shape=(None, 40, 40, 512)

    h5 = keras.layers.MaxPool2D(strides=(2, 2))(h4)
    hd5 = UnetConv2(h5, filters[4],n=n) #shape=(None, 20, 20, 1024)

    h1_PT_hd4 = keras.layers.MaxPool2D(strides=(8, 8))(h1)
    h1_PT_hd4 = conv_block(h1_PT_hd4, filters[0]) #shape=(None, 40, 40, 64

    h2_PT_hd4 = keras.layers.MaxPool2D(strides=(4, 4))(h2)
    h2_PT_hd4 = conv_block(h2_PT_hd4, filters[1]) #shape=(None, 40, 40, 128)

    h3_PT_hd4 = keras.layers.MaxPool2D(strides=(2, 2))(h3)
    h3_PT_hd4 = conv_block(h3_PT_hd4, filters[2]) # shape=(None, 40, 40, 256)

    h4_Cat_hd4=conv_block(h4, filters[3]) #shape=(None, 40, 40, 512)

    hd5_UT_hd4=keras.layers.UpSampling2D(size=(2,2))(hd5)
    hd5_UT_hd4=conv_block(hd5_UT_hd4,filters[4]) #shape=(None, 40, 40, 1024)

    #fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
    hd4=keras.layers.concatenate([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4]) #shape=(None, 40, 40, 1984)
    hd4 = conv_block(hd4, UpChannels) #shape=(None, 40, 40, 320)

    #stage 3d
    h1_PT_hd3 = keras.layers.MaxPool2D(strides=(4, 4))(h1)
    h1_PT_hd3=conv_block(h1_PT_hd3, filters[0]) #shape=(None, 80, 80, 64)

    h2_PT_hd3 = keras.layers.MaxPool2D(strides=(2, 2))(h2)
    h2_PT_hd3 = conv_block(h2_PT_hd3, filters[1]) #shape=(None, 80, 80, 128)

    h3_Cat_hd3=conv_block(h3, filters[2]) #shape=(None, 80, 80, 256)

    hd4_UT_hd3=keras.layers.UpSampling2D(size=(2,2))(hd4)
    hd4_UT_hd3 =conv_block(hd4_UT_hd3,UpChannels) #shape=(None, 80, 80, 320)

    hd5_UT_hd3=keras.layers.UpSampling2D(size=(4,4))(hd5)
    hd5_UT_hd3 = conv_block(hd5_UT_hd3, UpChannels) #shape=(None, 80, 80, 320)

    # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
    hd3=keras.layers.concatenate([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3])
    hd3 = conv_block(hd3, UpChannels) #shape=(None, 80, 80, 320)

    #stage 2d
    h1_PT_hd2 =  keras.layers.MaxPool2D(strides=(2, 2))(h1)
    h1_PT_hd2 = conv_block(h1_PT_hd2, filters[0]) #shape=(None, 160, 160, 64)

    h2_Cat_hd2=conv_block(h2, filters[1]) #shape=(None, 160, 160, 128)

    hd3_UT_hd2=keras.layers.UpSampling2D(size=(2,2))(hd3)
    hd3_UT_hd2 = conv_block(hd3_UT_hd2, UpChannels) #shape=(None, 160, 160, 320)

    hd4_UT_hd2 = keras.layers.UpSampling2D(size=(4,4))(hd4)
    hd4_UT_hd2 = conv_block(hd4_UT_hd2, UpChannels) # shape=(None, 160, 160, 320)

    hd5_UT_hd2 = keras.layers.UpSampling2D(size=(8, 8))(hd5)
    hd5_UT_hd2 = conv_block(hd5_UT_hd2, UpChannels)  # shape=(None, 160, 160, 320)

    ## fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
    hd2=keras.layers.concatenate([h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2])
    hd2 = conv_block(hd2, UpChannels) #shape=(None, 160, 160, 320)

    #stage 1d
    h1_Cat_hd1 = conv_block(h1, filters[0]) # shape=(None, 320, 320, 64)

    hd2_UT_hd1 =keras.layers.UpSampling2D(size=(2,2))(hd2)
    hd2_UT_hd1 = conv_block(hd2_UT_hd1, UpChannels) #shape=(None, 320, 320, 320)

    hd3_UT_hd1=keras.layers.UpSampling2D(size=(4,4))(hd3)
    hd3_UT_hd1 = conv_block(hd3_UT_hd1, UpChannels) #shape=(None, 320, 320, 320)

    hd4_UT_hd1=keras.layers.UpSampling2D(size=(8,8))(hd4)
    hd4_UT_hd1 = conv_block(hd4_UT_hd1, UpChannels) #shape=(None, 320, 320, 320)

    hd5_UT_hd1 = keras.layers.UpSampling2D(size=(16, 16))(hd5)
    hd5_UT_hd1 = conv_block(hd5_UT_hd1, UpChannels)  # shape=(None, 320, 320, 320)

    #fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
    hd1 = keras.layers.concatenate([h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1])
    hd1 = conv_block(hd1, UpChannels)  # shape=(None, 320, 320, 320)

    d5 = keras.layers.Conv2D(1,kernel_size=3,activation=None,padding='same',use_bias=False)(hd5)
    d5 = keras.layers.UpSampling2D(size=(16, 16)  )(d5)
    d55=keras.layers.Activation('sigmoid',name='d5')(d5)

    d4 =keras.layers.Conv2D(1,kernel_size=3,activation=None,padding='same',use_bias=False)(hd4)
    d4 = keras.layers.UpSampling2D(size=(8, 8)  )(d4)
    d44 = keras.layers.Activation('sigmoid',name='d4')(d4)

    d3 = keras.layers.Conv2D(1,kernel_size=3,activation=None,padding='same',use_bias=False)(hd3)
    d3 = keras.layers.UpSampling2D(size=(4, 4)  )(d3)
    d33 = keras.layers.Activation('sigmoid',name='d3')(d3)

    d2 = keras.layers.Conv2D(1,kernel_size=3,activation=None,padding='same',use_bias=False)(hd2)
    d2 = keras.layers.UpSampling2D(size=(2, 2)  )(d2)
    d22 = keras.layers.Activation('sigmoid',name='d2')(d2)

    d1 = keras.layers.Conv2D(1,kernel_size=3,activation=None,padding='same',use_bias=False)(hd1)
    #
    d11=keras.layers.Activation('sigmoid',name='d1')(d1)
    d = keras.layers.average([d1, d2, d3, d4, d5])
    d = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d)
    d = keras.layers.Activation('sigmoid', name='d')(d)
    model=keras.models.Model(inputs=inputs,outputs=[d,d11,d22,d33,d44,d55])
    return model

if __name__ == '__main__':
    model=Unet3(input_shape=(256,256,3))
    model.summary()
    keras.utils.plot_model(model,'Unet3-2.png',show_shapes=True)

    #cls_branch=keras.layers.Dropout(0.5)(hd5)
    #cls_branch =keras.layers.Conv2D(2,kernel_size=1,padding='same')(cls_branch)
    #cls_branch=keras.layers.MaxPool2D()(cls_branch )
    #cls_branch =keras.layers.Activation('sigmoid')(cls_branch ) #shape=(None, 10, 10, 1024)






