""" Deeplabv3+ model for Keras.
This model is based on TF repo:
https://github.com/tensorflow/models/tree/master/research/deeplab
On Pascal VOC, original model gets to 84.56% mIOU
MobileNetv2 backbone is based on this repo:
https://github.com/JonathanCMitchell/mobilenet_v2_keras
# Reference
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Xception: Deep Learning with Depthwise Separable Convolutions]
    (https://arxiv.org/abs/1610.02357)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
"""
#https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras import backend as K
#from tensorflow.python.keras.applications.imagenet_utils import preprocess_input

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_X_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
WEIGHTS_PATH_MOBILE_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5"


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = keras.layers.Activation(tf.nn.relu)(x)
    x = keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = keras.layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = keras.layers.Activation(tf.nn.relu)(x)
    x = keras.layers.Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = keras.layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = keras.layers.ReLU()(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return keras.layers.Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
        return keras.layers.Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = keras.layers.BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = keras.layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = keras.layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = int(inputs.shape[-1])  # inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = keras.layers.Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name=prefix + 'expand_BN')(x)
        x = keras.layers.Activation(tf.nn.relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,name=prefix + 'depthwise_BN')(x)

    x = keras.layers.Activation(tf.nn.relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = keras.layers.Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return keras.layers.Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def Deeplabv3(input_shape=(512, 512, 3), classes=21, backbone='mobilenetv2',
              OS=16, alpha=1., activation=None):
    """ Instantiates the Deeplabv3+ architecture
    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
    # Arguments
        weights: one of 'pascal_voc' (pre-trained on pascal voc),
            'cityscapes' (pre-trained on cityscape) or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
        classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
            If number of classes not aligned with the weights used, last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        activation: optional activation to add to the top of the network.
            One of 'softmax', 'sigmoid' or None
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
    """



    if not (backbone in {'xception', 'mobilenetv2'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`  or `mobilenetv2` ')


    img_input = keras.layers.Input(shape=input_shape)

    if backbone == 'xception':
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        x = keras.layers.Conv2D(32, (3, 3), strides=(2, 2),name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
        x = keras.layers.BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = keras.layers.ReLU()(x) #256 256 32

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = keras.layers.BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = keras.layers.ReLU()(x)

        x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',skip_connection_type='conv', stride=2,depth_activation=False) #128 128 128
        x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',skip_connection_type='conv', stride=2,depth_activation=False, return_skip=True) #64 64 256,128, 128 256

        x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',skip_connection_type='conv', stride=entry_block3_stride,depth_activation=False)#32 32 728
        for i in range(16):
            x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),skip_connection_type='sum', stride=1, rate=middle_block_rate,depth_activation=False)
            # 32 32 728

        x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1', skip_connection_type='conv', stride=1, rate=exit_block_rates[0], depth_activation=False)
        x7 = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',skip_connection_type='none', stride=1, rate=exit_block_rates[1],depth_activation=True) #32 32 2048

    else:
        OS = 8
        first_block_filters = _make_divisible(32 * alpha, 8)
        x0 = keras.layers.Conv2D(first_block_filters, kernel_size=3,strides=(2, 2), padding='same',use_bias=False, name='Conv')(img_input)
        x0 = keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x0)
        x0 = keras.layers.Activation(tf.nn.relu6, name='Conv_Relu6')(x0) # 256 256 32

        x1 = _inverted_res_block(x0, filters=16, alpha=alpha, stride=1,expansion=1, block_id=0, skip_connection=False) # 256 256 16

        x2 = _inverted_res_block(x1, filters=24, alpha=alpha, stride=2,expansion=6, block_id=1, skip_connection=False) #128 128 24
        x2 = _inverted_res_block(x2, filters=24, alpha=alpha, stride=1,expansion=6, block_id=2, skip_connection=True) #128 128 24

        x3 = _inverted_res_block(x2, filters=32, alpha=alpha, stride=2,expansion=6, block_id=3, skip_connection=False) #64 64 32
        x3 = _inverted_res_block(x3, filters=32, alpha=alpha, stride=1,expansion=6, block_id=4, skip_connection=True) #64 64 32
        x3 = _inverted_res_block(x3, filters=32, alpha=alpha, stride=1,expansion=6, block_id=5, skip_connection=True) #64 64 32

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x4 = _inverted_res_block(x3, filters=64, alpha=alpha, stride=1,  # 1!
                                expansion=6, block_id=6, skip_connection=False) #64 64 64
        x4 = _inverted_res_block(x4, filters=64, alpha=alpha, stride=1, rate=2,expansion=6, block_id=7, skip_connection=True) #64 64 64
        x4 = _inverted_res_block(x4, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=8, skip_connection=True) #64 64 64
        x4 = _inverted_res_block(x4, filters=64, alpha=alpha, stride=1, rate=2,expansion=6, block_id=9, skip_connection=True)#64 64 64

        x5 = _inverted_res_block(x4, filters=96, alpha=alpha, stride=2, rate=2, expansion=6, block_id=10, skip_connection=False) #32 32 96
        x5 = _inverted_res_block(x5, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=11, skip_connection=True) #64 64 96
        x5 = _inverted_res_block(x5, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=12, skip_connection=True) #64 64 96

        x6 = _inverted_res_block(x5, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                expansion=6, block_id=13, skip_connection=False) #64 64 160
        x6 = _inverted_res_block(x6, filters=160, alpha=alpha, stride=1, rate=4,expansion=6, block_id=14, skip_connection=True) #64 64 160
        x6 = _inverted_res_block(x6, filters=160, alpha=alpha, stride=1, rate=4,expansion=6, block_id=15, skip_connection=True) #64 64 160

        x7 = _inverted_res_block(x6, filters=320, alpha=alpha, stride=1, rate=4,expansion=6, block_id=16, skip_connection=False) #64 64 320

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    shape_before = tf.shape(x7)
    b4 = keras.layers.GlobalAveragePooling2D()(x7)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = keras.layers.Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = keras.layers.Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = keras.layers.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = keras.layers.ReLU()(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = keras.backend.int_shape(x7)
    b4 = keras.layers.Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],method='bilinear', align_corners=True))(b4) #64 64 256
    # simple 1x1
    b0 = keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x7) #32 32 256
    b0 = keras.layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = keras.layers.ReLU()(b0) #64 64 256

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x7, 256, 'aspp1',rate=atrous_rates[0], depth_activation=True, epsilon=1e-5) #32 32 256
        # rate = 12 (24)
        b2 = SepConv_BN(x7, 256, 'aspp2',rate=atrous_rates[1], depth_activation=True, epsilon=1e-5) #32 32 256
        # rate = 18 (36)
        b3 = SepConv_BN(x7, 256, 'aspp3', rate=atrous_rates[2], depth_activation=True, epsilon=1e-5) #32 32 256

        # concatenate ASPP branches & project
        x = keras.layers.Concatenate()([b4, b0, b1, b2, b3]) # 32 32 1280
    else:
        x = keras.layers.Concatenate()([b4, b0])

    x = keras.layers.Conv2D(256, (1, 1), padding='same',use_bias=False, name='concat_projection')(x)
    x = keras.layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = keras.layers.ReLU()(x) #64 64 256
    x = keras.layers.Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        #skip_size = keras.backend.int_shape(skip1)
        #x = keras.layers.Lambda(lambda xx: tf.compat.v1.image.resize(xx, skip_size[1:3], method='bilinear', align_corners=True))(x) #128 128 256
        x=keras.layers.UpSampling2D(size=(4,4))(x)

        dec_skip1 = keras.layers.Conv2D(48, (1, 1), padding='same',use_bias=False, name='feature_projection0')(skip1) # 128 128 48
        dec_skip1 = keras.layers.BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = keras.layers.Activation(tf.nn.relu)(dec_skip1)
        x = keras.layers.Concatenate()([x, dec_skip1]) # 128 128 304
        x = SepConv_BN(x, 256, 'decoder_conv0',depth_activation=True, epsilon=1e-5) #128 128 256
        x = SepConv_BN(x, 256, 'decoder_conv1',depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    print(x.shape)
    x=keras.layers.concatenate([x,x6])
    x = keras.layers.Conv2DTranspose(256, kernel_size=(1, 1), strides=(2, 2), padding='same')(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.ReLU()(x)
    x = keras.layers.concatenate([x, x4])

    x = keras.layers.Conv2DTranspose(256, kernel_size=(1, 1), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.concatenate([x, x2])
    x = keras.layers.Conv2DTranspose(128, kernel_size=(1, 1), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)


    x = keras.layers.Conv2D(classes, (3, 3), padding='same', name='last_layer')(x) #(None, 64, 64, 1)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    #size_before3 = tf.keras.backend.int_shape(img_input)
    print(x.shape)
    #x = keras.layers.Lambda(lambda xx: tf.compat.v1.image.resize(xx,size_before3[1:3], method='bilinear', align_corners=True))(x)

    #x=keras.layers.UpSampling2D(size=(8,8),interpolation='bilinear')(x)
    x=keras.layers.Conv2DTranspose(classes,kernel_size=(3,3),strides=(2,2),padding='same')(x)


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.

    if activation in {'softmax', 'sigmoid'}:
        d1 = keras.layers.Activation(activation,name='d1')(x)

    model = keras.models.Model(img_input, d1, name='deeplabv3plus')
    '''
    d2=keras.layers.UpSampling2D(size=(8,8))(x7)
    d2=keras.layers.Conv2D(classes, (1, 1), padding='same')(d2)
    d22 = keras.layers.Activation(activation, name='d2')(d2)

    #d3 = keras.layers.UpSampling2D(size=(4, 4))(x6)
    #d3 = Conv2D(classes, (1, 1), padding='same')(d3)

    d3 = keras.layers.UpSampling2D(size=(8, 8))(x5)
    d3 = keras.layers.Conv2D(classes, (1, 1), padding='same')(d3)
    d33 = keras.layers.Activation(activation, name='d3')(d3)

    d4 = keras.layers.UpSampling2D(size=(4, 4))(x2)
    d4 = keras.layers.Conv2D(classes, (1, 1), padding='same')(d4)
    d44 = keras.layers.Activation(activation, name='d4')(d4)

    d5 = keras.layers.UpSampling2D(size=(2, 2))(x0)
    d5 = keras.layers.Conv2D(classes, (1, 1), padding='same')(d5)
    d55 = keras.layers.Activation(activation, name='d5')(d5)

    d=keras.layers.concatenate([x,d2,d3,d4,d5])
    d = keras.layers.Conv2D(classes, (1, 1), padding='same')(d)
    d= keras.layers.Activation(activation, name='d')(d)
    model1=keras.models.Model(inputs=img_input,outputs=[d,d1,d22,d33,d44,d55])
    '''
    return model

if __name__ == '__main__':
    model=Deeplabv3(input_shape=(512, 512, 3), classes=1, backbone='mobilenetv2',OS=16, alpha=1., activation='sigmoid')
    model.summary()
    '''
    mobilenetv2
    Total params: 2,209,355
    Trainable params: 2,176,267
    Non-trainable params: 33,088
        
    Total params: 41,255,131
   Trainable params: 41,052,331
   Non-trainable params: 202,800
    '''