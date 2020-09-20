from tensorflow.keras import backend as K
from tensorflow import keras

"""Building Block Functions"""


def se_block(inputs, reduction=16):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = inputs.shape[channel_axis]
    nn = keras.layers.GlobalAveragePooling2D()(inputs)
    nn = keras.layers.Reshape((1, 1, filters))(nn)
    nn = keras.layers.Conv2D(filters // reduction, kernel_size=1)(nn)
    # nn = keras.layers.PReLU(shared_axes=[1, 2])(nn)
    nn = keras.layers.ReLU()(nn)
    nn = keras.layers.Conv2D(filters, kernel_size=1, activation="sigmoid")(nn)
    nn = keras.layers.Multiply()([inputs, nn])
    return nn


def se_block_2(inputs, reduction=16):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = inputs.shape[channel_axis]
    se = keras.layers.GlobalAveragePooling2D()(inputs)
    se = keras.layers.Dense(filters // reduction, activation="PReLU", use_bias=False)(se)
    se = keras.layers.Dense(filters, activation="sigmoid", use_bias=False)(se)
    # if K.image_data_format() == 'channels_first':
    #     se = Permute((3, 1, 2))(se)
    x = keras.layers.Multiply()([inputs, se])
    return x


def conv_block(inputs, filters, kernel_size, strides, padding):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    Z = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(inputs)
    Z = keras.layers.BatchNormalization(axis=channel_axis)(Z)
    A = keras.layers.PReLU(shared_axes=[1, 2])(Z)
    return A


def separable_conv_block(inputs, filters, kernel_size, strides):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    Z = keras.layers.SeparableConv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(inputs)
    Z = keras.layers.BatchNormalization(axis=channel_axis)(Z)
    # A = keras.layers.PReLU(shared_axes=[1, 2])(Z)
    A = keras.layers.ReLU()(Z)
    return A


def bottleneck(inputs, filters, kernel, t, s, r=False, se=False):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    Z1 = conv_block(inputs, tchannel, 1, 1, "same")
    Z1 = keras.layers.DepthwiseConv2D(kernel, strides=s, padding="same", depth_multiplier=1, use_bias=False)(Z1)
    Z1 = keras.layers.BatchNormalization(axis=channel_axis)(Z1)
    # A1 = keras.layers.PReLU(shared_axes=[1, 2])(Z1)
    A1 = keras.layers.ReLU()(Z1)
    Z2 = keras.layers.Conv2D(filters, 1, strides=1, padding="same", use_bias=False)(A1)
    Z2 = keras.layers.BatchNormalization(axis=channel_axis)(Z2)
    if se:
        Z2 = se_block(Z2)
    if r:
        Z2 = keras.layers.add([Z2, inputs])
    return Z2


def inverted_residual_block(inputs, filters, kernel, t, strides, n, se=False):
    Z = bottleneck(inputs, filters, kernel, t, strides, se=se)
    for i in range(1, n):
        Z = bottleneck(Z, filters, kernel, t, 1, True, se=se)
    return Z


def linear_GD_conv_block(inputs, kernel_size, strides):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    Z = keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding="valid", depth_multiplier=1, use_bias=False)(inputs)
    Z = keras.layers.BatchNormalization(axis=channel_axis)(Z)
    return Z


def UMFacenet(input_shape=(256, 256, 3), dropout=0.5, filters=[64, 128, 256, 512], use_se=True):
    # channel_axis =  -1
    X = keras.layers.Input(shape=input_shape)
    M0 = conv_block(X, filters[0], 3, 2, "same")  # 128
    M = separable_conv_block(M0, filters[0], 3, 1)  # 128
    M1 = inverted_residual_block(M, filters[0], 3, t=2, strides=2, n=5, se=use_se)  # 64
    if dropout > 0.:
        M1 = keras.layers.Dropout(dropout)(M1)
    M2 = inverted_residual_block(M1, filters[1], 3, t=4, strides=2, n=1, se=use_se)  # 32
    M2 = inverted_residual_block(M2, filters[1], 3, t=2, strides=1, n=6, se=use_se)  # 32
    if dropout > 0.:
        M2 = keras.layers.Dropout(dropout)(M2)

    M3 = inverted_residual_block(M2, filters[2], 3, t=4, strides=2, n=1, se=use_se)  # 16
    M3 = inverted_residual_block(M3, filters[2], 3, t=2, strides=1, n=2, se=use_se)  # 16

    if dropout > 0.:
        M3 = keras.layers.Dropout(dropout)(M3)

    M4 = inverted_residual_block(M3, filters[3], 3, t=4, strides=2, n=1, se=use_se)  # 8
    M4 = inverted_residual_block(M4, filters[3], 3, t=2, strides=1, n=2, se=use_se)

    # up4=keras.layers.UpSampling2D(size=(2,2))(M4)
    up4 = keras.layers.Conv2DTranspose(filters[3], (3, 3), strides=(2, 2), padding="same", name='up2_trans4d')(M4)
    up4d = keras.layers.concatenate([up4, M3])
    # up4d=inverted_residual_block(up4d, filters[3], 3, t=2, strides=1, n=2, se=use_se)
    up4d = inverted_residual_block(up4d, filters[3], 3, t=4, strides=1, n=1, se=use_se)  # 16

    # up3=keras.layers.UpSampling2D(size=(2,2))(up4d)
    up3 = keras.layers.Conv2DTranspose(filters[2], (3, 3), strides=(2, 2), padding="same", name='up3_trans2d')(up4d)
    up3d = keras.layers.concatenate([up3, M2])
    # up3d=inverted_residual_block(up3d, filters[2], 3, t=2, strides=1, n=2, se=use_se) #32
    up3d = inverted_residual_block(up3d, filters[2], 3, t=4, strides=1, n=1, se=use_se)

    # up2=keras.layers.UpSampling2D(size=(2,2))(up3d)
    up2 = keras.layers.Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", name='up2_trans2d')(up3d)
    up2d = keras.layers.concatenate([up2, M1])
    # up2d= inverted_residual_block(up2d ,filters[1], 3, t=2, strides=1, n=6, se=use_se) #64
    up2d = inverted_residual_block(up2d, filters[1], 3, t=4, strides=1, n=1, se=use_se)

    # up1 = keras.layers.UpSampling2D(size=(2, 2))(up2d) #128
    up1 = keras.layers.Conv2DTranspose(filters[0], (3, 3), strides=(2, 2), padding="same", name='up1_trans2d')(up2d)
    up1d = keras.layers.concatenate([up1, M])
    up1d = inverted_residual_block(up1d, filters[0], 3, t=2, strides=1, n=5, se=use_se)
    up1d = separable_conv_block(up1d, filters[0], 3, 1)
    up1d = conv_block(up1d, filters[0], 3, 1, "same")

    d1 = keras.layers.Conv2DTranspose(filters[0], (3, 3), strides=(2, 2), padding="same", name='d1_trans2d')(up1d)
    d1 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d1)
    d11 = keras.layers.Activation('sigmoid', name='d1')(d1)

    # d2 = keras.layers.Conv2DTranspose(filters[1], (3, 3), strides=(4, 4), padding="same",name='d2_trans2d_1')(up2d)
    d2 = keras.layers.UpSampling2D(size=(4, 4))(up2d)
    # d2 = keras.layers.Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", name='d2_trans2d_2')(d2)
    d2 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d2)
    d22 = keras.layers.Activation('sigmoid', name='d2')(d2)

    # d3 = keras.layers.Conv2DTranspose(filters[2], (3, 3), strides=(8, 8), padding="same",name='d3_trans2d')(up3d)
    d3 = keras.layers.UpSampling2D(size=(8, 8))(up3d)
    d3 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d3)
    d33 = keras.layers.Activation('sigmoid', name='d3')(d3)

    # d4 = keras.layers.Conv2DTranspose(filters[3], (3, 3), strides=(16, 16), padding="same")(up4d)
    d4 = keras.layers.UpSampling2D(size=(16, 16))(up4d)
    d4 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d4)
    d44 = keras.layers.Activation('sigmoid', name='d4')(d4)

    d5 = keras.layers.Conv2DTranspose(filters[0], (3, 3), strides=(2, 2), padding="same")(M)
    d5 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d5)
    d55 = keras.layers.Activation('sigmoid', name='d5')(d5)

    d = keras.layers.concatenate([d1, d2, d3, d4, d5, X])
    d = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d)
    d = keras.layers.Activation('sigmoid', name='d')(d)

    model = keras.models.Model(inputs=X, outputs=[d, d11, d22, d33, d44, d55])
    return model


if __name__ == '__main__':
    '''
    Total params: 12,795,384
    Trainable params: 12,709,624
    Non-trainable params: 85,760

    Total params: 4,015,016
    Trainable params: 3,972,136
    Non-trainable params: 42,880

    Total params: 3,415,976
   Trainable params: 3,373,096
   Non-trainable params: 42,880
    '''
    # model=UMFacenet(input_shape=(256,256,3))
    model = UMFacenet(input_shape=(256, 256, 3), filters=[32, 64, 128, 256], use_se=False)
    model.summary()
    keras.utils.plot_model(model, 'UMFacenetS0.png', show_shapes=True)