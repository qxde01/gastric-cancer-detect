# https://github.com/godofpdog/MobileNetV3_keras/blob/master/src/MobileNet_V3.py
import tensorflow as tf

from tensorflow import keras
import tensorflow.keras.backend as K

""" Define layers block functions """


def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6


# ** update custom Activate functions
keras.utils.get_custom_objects().update({'custom_activation': keras.layers.Activation(Hswish)})


def __conv2d_block(_inputs, filters, kernel, strides, is_use_bias=False, padding='same', activation='RE', name=None):
    x = keras.layers.Conv2D(filters, kernel, strides=strides, padding=padding, use_bias=is_use_bias, kernel_initializer='he_normal')(_inputs)
    x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    if activation == 'RE':
        x = keras.layers.ReLU(name=name)(x)
    elif activation == 'HS':
        x = keras.layers.Activation(Hswish, name=name)(x)
    else:
        raise NotImplementedError
    return x


def __depthwise_block(_inputs, kernel=(3, 3), strides=(1, 1), activation='RE', is_use_se=True, num_layers=0):
    x = keras.layers.DepthwiseConv2D(kernel_size=kernel, strides=strides, depth_multiplier=1, padding='same', kernel_initializer='he_normal')(_inputs)
    x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    if is_use_se:
        x = __se_block(x)
    if activation == 'RE':
        x = keras.layers.ReLU()(x)
    elif activation == 'HS':
        x = keras.layers.Activation(Hswish)(x)
    else:
        raise NotImplementedError
    return x


def __global_depthwise_block(_inputs):
    print(_inputs)
    # assert _inputs._keras_shape[1] == _inputs._keras_shape[2]
    assert _inputs.shape[1] == _inputs.shape[2]
    # kernel_size = _inputs._keras_shape[1]
    kernel_size = int(_inputs.shape[1])
    x = keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(1, 1), depth_multiplier=1, padding='valid', kernel_initializer='he_normal')(_inputs)
    return x


def __se_block(_inputs, ratio=4, pooling_type='avg'):
    # print('============'*10)
    # print(_inputs,_inputs.shape,_inputs.shape[-1])
    # filters = _inputs._keras_shape[-1]
    filters = int(_inputs.shape[-1])
    se_shape = (1, 1, filters)
    if pooling_type == 'avg':
        se = keras.layers.GlobalAveragePooling2D()(_inputs)
    elif pooling_type == 'depthwise':
        se = __global_depthwise_block(_inputs)
    else:
        raise NotImplementedError
    se = keras.layers.Reshape(se_shape)(se)
    se = keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = keras.layers.Dense(filters, activation='hard_sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return keras.layers.multiply([_inputs, se])


def __bottleneck_block(_inputs, out_dim, kernel, strides, expansion_dim, is_use_bias=False, shortcut=True, is_use_se=True, activation='RE', num_layers=0, *args):
    with tf.name_scope('bottleneck_block'):
        # ** to high dim
        bottleneck_dim = expansion_dim

        # ** pointwise conv
        x = __conv2d_block(_inputs, bottleneck_dim, kernel=(1, 1), strides=(1, 1), is_use_bias=is_use_bias, activation=activation)

        # ** depthwise conv
        x = __depthwise_block(x, kernel=kernel, strides=strides, is_use_se=is_use_se, activation=activation, num_layers=num_layers)

        # ** pointwise conv
        x = keras.layers.Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

        if shortcut and strides == (1, 1):
            in_dim = K.int_shape(_inputs)[-1]
            if in_dim != out_dim:
                ins = keras.layers.Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(_inputs)
                x = keras.layers.Add()([x, ins])
            else:
                x = keras.layers.Add()([x, _inputs])
    return x


def build_mobilenet_v3(input_shape=(224, 224, 3), model_type='large'):
    # ** input layer
    inputs = keras.layers.Input(shape=input_shape)

    # ** feature extraction layers
    # cifar100 strides=(2, 2)=>(1,1)
    net = __conv2d_block(inputs, 16, kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS')  # shape=(None, 112, 112, 16)

    if model_type == 'large':
        config_list = large_config_list
    elif model_type == 'small':
        config_list = small_config_list
    else:
        raise NotImplementedError

    net0 = __bottleneck_block(net, *config_list[0])  # shape=(None, 56, 56, 16)
    net1 = __bottleneck_block(net0, *config_list[1])  # shape=(None, 28, 28, 24)
    net2 = __bottleneck_block(net1, *config_list[2])  # shape=(None, 28, 28, 24)
    net3 = __bottleneck_block(net2, *config_list[3])  # shape=(None, 28, 28, 40)
    net4 = __bottleneck_block(net3, *config_list[4])  # shape=(None, 28, 28, 40)
    net5 = __bottleneck_block(net4, *config_list[5])  # shape=(None, 28, 28, 40)
    net6 = __bottleneck_block(net5, *config_list[6])  # shape=(None, 28, 28, 48)
    net7 = __bottleneck_block(net6, *config_list[7])  # shape=(None, 28, 28, 48)
    net8 = __bottleneck_block(net7, *config_list[8])  # shape=(None, 14, 14, 96)
    net9 = __bottleneck_block(net8, *config_list[9])  # shape=(None, 14, 14, 96)
    net10 = __bottleneck_block(net9, *config_list[10])  # shape=(None, 14, 14, 96)

    # ** final layers
    net11 = __conv2d_block(net10, 480, kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='HS', name='output_map')

    net12 = keras.layers.Conv2D(640, (1, 1), strides=(1, 1), padding='valid', use_bias=True, kernel_initializer='he_normal')(net11)

    net12 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(net12)
    net12 = keras.layers.Activation(Hswish)(net12)

    up5 = keras.layers.UpSampling2D(size=(2, 2))(net12)
    # up5=keras.layers.Conv2DTranspose(int(net12.shape[-1]),kernel_size=3,strides=(2,2),padding='same')(net12)
    up5d = keras.layers.concatenate([up5, net7])
    up5d = __bottleneck_block(up5d, *config_list[7])  # shape=(None, 28, 28, 48)
    up5d = __bottleneck_block(up5d, *config_list[6])

    up4 = keras.layers.UpSampling2D(size=(2, 2))(up5d)  # shape=(None, 56, 56, 48)
    # up4=keras.layers.Conv2DTranspose(int(up5d.shape[-1]), kernel_size=3, strides=(2, 2), padding='same')(up5d)
    up4d = keras.layers.concatenate([up4, net0])  # shape=(None, 56, 56, 64)
    up4d = __bottleneck_block(up4d, *config_list[5])  # shape=(None, 56, 56, 40)
    up4d = __bottleneck_block(up4d, *config_list[4])

    up3 = keras.layers.UpSampling2D(size=(2, 2))(up4d)  # shape=(None, 112, 112, 40)
    up3d = keras.layers.concatenate([up3, net])  # shape=(None, 112, 112, 56)
    up3d = __bottleneck_block(up3d, *config_list[3])
    up3d = __bottleneck_block(up3d, *config_list[2])  # shape=(None, 112, 112, 24)

    up2 = keras.layers.UpSampling2D(size=(2, 2))(up3d)  # shape=(None, 224, 224, 24)
    d1 = __conv2d_block(up2, 16, kernel=(3, 3), strides=(1, 1), is_use_bias=False, padding='same', activation='HS')
    d1 = keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same', use_bias=True, kernel_initializer='he_normal')(d1)
    d11 = keras.layers.Activation('sigmoid', name='d1')(d1)

    d2 = keras.layers.UpSampling2D(size=(4, 4))(up4d)
    # d2 = keras.layers.Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", name='d2_trans2d_2')(d2)
    d2 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d2)
    d22 = keras.layers.Activation('sigmoid', name='d2')(d2)

    d3 = keras.layers.UpSampling2D(size=(8, 8))(up5d)
    # d2 = keras.layers.Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", name='d2_trans2d_2')(d2)
    d3 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d3)
    d33 = keras.layers.Activation('sigmoid', name='d3')(d3)

    d4 = keras.layers.UpSampling2D(size=(16, 16))(net11)
    d4 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d4)
    d44 = keras.layers.Activation('sigmoid', name='d4')(d4)

    d5 = keras.layers.UpSampling2D(size=(2, 2))(net)
    d5 = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d5)
    d55 = keras.layers.Activation('sigmoid', name='d5')(d5)

    d = keras.layers.concatenate([d1, d2, d3, d4, d5, inputs])
    d = keras.layers.Conv2D(1, kernel_size=3, activation=None, padding='same', use_bias=False)(d)
    d = keras.layers.Activation('sigmoid', name='d')(d)

    model = keras.models.Model(inputs=inputs, outputs=[d, d11, d22, d33, d44, d55])
    return model


""" define bottleneck structure """
# **
# **
global large_config_list
global small_config_list

large_config_list = [[16, (3, 3), (1, 1), 16, False, False, False, 'RE', 0],
                     [24, (3, 3), (2, 2), 64, False, False, False, 'RE', 1],  # strides=(2, 2)=>(1,1)
                     [24, (3, 3), (1, 1), 72, False, True, False, 'RE', 2],
                     [40, (5, 5), (2, 2), 72, False, False, True, 'RE', 3],
                     [40, (5, 5), (1, 1), 120, False, True, True, 'RE', 4],
                     [40, (5, 5), (1, 1), 120, False, True, True, 'RE', 5],
                     [80, (3, 3), (2, 2), 240, False, False, False, 'HS', 6],
                     [80, (3, 3), (1, 1), 200, False, True, False, 'HS', 7],
                     [80, (3, 3), (1, 1), 184, False, True, False, 'HS', 8],
                     [80, (3, 3), (1, 1), 184, False, True, False, 'HS', 9],
                     [112, (3, 3), (1, 1), 480, False, False, True, 'HS', 10],
                     [112, (3, 3), (1, 1), 672, False, True, True, 'HS', 11],
                     [160, (5, 5), (1, 1), 672, False, False, True, 'HS', 12],
                     [160, (5, 5), (2, 2), 672, False, True, True, 'HS', 13],
                     [160, (5, 5), (1, 1), 960, False, True, True, 'HS', 14]]

small_config_list = [[16, (3, 3), (2, 2), 16, False, False, True, 'RE', 0],
                     [24, (3, 3), (2, 2), 72, False, False, False, 'RE', 1],
                     [24, (3, 3), (1, 1), 88, False, True, False, 'RE', 2],
                     [40, (5, 5), (1, 1), 96, False, False, True, 'HS', 3],
                     [40, (5, 5), (1, 1), 240, False, True, True, 'HS', 4],
                     [40, (5, 5), (1, 1), 240, False, True, True, 'HS', 5],
                     [48, (5, 5), (1, 1), 120, False, False, True, 'HS', 6],
                     [48, (5, 5), (1, 1), 144, False, True, True, 'HS', 7],
                     [96, (5, 5), (2, 2), 288, False, False, True, 'HS', 8],
                     [96, (5, 5), (1, 1), 576, False, True, True, 'HS', 9],
                     [96, (5, 5), (1, 1), 576, False, True, True, 'HS', 10]]


# def MobileNetV3Large(include_top=True,input_shape=(416,416,3), num_classes=10, pooling='avg'):
#    return build_mobilenet_v3(input_shape=input_shape, num_classes=num_classes, model_type='large', pooling_type=pooling, include_top=include_top)

def MobileNetV3Small(input_shape=(224, 224, 3)):
    return build_mobilenet_v3(input_shape=input_shape, model_type='small')


""" build MobileNet V3 model """
if __name__ == '__main__':
    # MobileNetV3Large,5,246,972     5,252,092
    # MobileNetV3Small,3,068,796    3,073,916
    '''
    Total params: 2,109,657
    Trainable params: 2,092,521
    Non-trainable params: 17,136
    18,153,177
    3,427,081
    3,406,297
    1,937,177

    '''
    model = build_mobilenet_v3(input_shape=(224, 224, 3), model_type='small')
    print(model.summary())
    keras.utils.plot_model(model, 'MobileNetV3small.png', show_shapes=True)
    # print(model.layers)
