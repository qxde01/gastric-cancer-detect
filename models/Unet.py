from tensorflow import keras
#https://github.com/zhixuhao/unet

def conv_block(inputs, filters, kernel_size, strides, padding='same'):
    Z = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(inputs)
    Z = keras.layers.BatchNormalization(axis=-1)(Z)
    A = keras.layers.PReLU(shared_axes=[1, 2])(Z)
    return A

def Unet(input_shape=(256, 256, 1)):
    inputs = keras.layers.Input(input_shape)
    conv1 =  conv_block(inputs, 32, 3, 1, padding='same')
    conv1 = conv_block(conv1, 32, 3, 1, padding='same')
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 64, 3, 1, padding='same')
    conv2 = conv_block(conv2, 64, 3, 1, padding='same')
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 128, 3, 1, padding='same')
    conv3 = conv_block(conv3, 128, 3, 1, padding='same')

    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, 256, 3, 1, padding='same')
    conv4 = conv_block(conv4, 256, 3, 1, padding='same')
    drop4 = keras.layers.Dropout(0.5)(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = conv_block(pool4, 512, 3, 1, padding='same')
    conv5 = conv_block(conv5, 512, 3, 1, padding='same')

    drop5 = keras.layers.Dropout(0.5)(conv5)

    up6 = conv_block(keras.layers.UpSampling2D(size=(2, 2))(drop5), 256, 3, 1, padding='same')
    merge6 = keras.layers.concatenate([drop4, up6], axis=3)
    conv6 =conv_block(merge6, 256, 3, 1, padding='same')
    conv6 = conv_block(conv6, 256, 3, 1, padding='same')

    up7 = conv_block(keras.layers.UpSampling2D(size=(2, 2))(conv6), 128, 3, 1, padding='same')
    merge7 = keras.layers.concatenate([conv3, up7], axis=3)

    conv7 =conv_block(merge7, 128, 3, 1, padding='same')
    conv7 = conv_block(conv7, 128, 3, 1, padding='same')

    up8 = conv_block(keras.layers.UpSampling2D(size=(2, 2))(conv7), 64, 3, 1, padding='same')
    merge8 = keras.layers.concatenate([conv2, up8], axis=3)

    conv8 =conv_block(merge8, 64, 3, 1, padding='same')
    conv8 = conv_block(conv8, 64, 3, 1, padding='same')

    up9 =conv_block(keras.layers.UpSampling2D(size=(2, 2))(conv8), 32, 3, 1, padding='same')
    merge9 = keras.layers.concatenate([conv1, up9], axis=3)

    conv9 =  conv_block(merge9, 32, 3, 1, padding='same')
    conv9 = conv_block(conv9, 32, 3, 1, padding='same')
    conv9 = conv_block(conv9, 3, 3, 1, padding='same')
    mask = keras.layers.Conv2D(1, 3, activation='sigmoid',name='mask', padding='same')(conv9)

    #output=keras.layers.Flatten()(drop5)
    #output = keras.layers.Dense(1, activation='sigmoid', name='classify')(output)
    #model = keras.models.Model(inputs=inputs, outputs=[output,mask])
    model = keras.models.Model(inputs=inputs, outputs=mask)

    return model
