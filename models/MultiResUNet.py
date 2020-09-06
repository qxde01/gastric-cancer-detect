from tensorflow import keras
#https://github.com/nibtehaz/MultiResUNet

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = keras.layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
    if activation == None:
        return x
    x = keras.layers.Activation(activation, name=name)(x)
    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    Returns:
        [keras layer] -- [output layer]
    '''
    x = keras.layers.Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding, kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
    return x


def MultiResBlock(U, inp, alpha=1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    W = alpha * U
    shortcut = inp
    shortcut = conv2d_bn(shortcut, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1, activation=None, padding='same')
    conv3x3 = conv2d_bn(inp, int(W * 0.167), 3, 3, activation='relu', padding='same')
    conv5x5 = conv2d_bn(conv3x3, int(W * 0.333), 3, 3, activation='relu', padding='same')
    conv7x7 = conv2d_bn(conv5x5, int(W * 0.5), 3, 3, activation='relu', padding='same')
    out = keras.layers.concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = keras.layers.BatchNormalization(axis=3)(out)
    out = keras.layers.add([shortcut, out])
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.BatchNormalization(axis=3)(out)
    # out=Dropout(0.5)(out)
    return out


def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')
    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')
    out = keras.layers.add([shortcut, out])
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.BatchNormalization(axis=3)(out)
    for i in range(length - 1):
        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')
        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')
        out = keras.layers.add([shortcut, out])
        out = keras.layers.Activation('relu')(out)
        out = keras.layers.BatchNormalization(axis=3)(out)
    return out


def MultiResUnet(input_shape=(224, 224, 3), classes=1):
    '''
    MultiResUNet
    '''

    inputs = keras.layers.Input(shape=input_shape)

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32 * 2, pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32 * 2, 3, mresblock2)

    mresblock3 = MultiResBlock(32 * 4, pool2)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32 * 4, 2, mresblock3)

    mresblock4 = MultiResBlock(32 * 8, pool3)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32 * 8, 1, mresblock4)

    mresblock5 = MultiResBlock(32 * 16, pool4)

    up6 = keras.layers.concatenate([keras.layers.Conv2DTranspose(32 * 8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32 * 8, up6)

    up7 = keras.layers.concatenate([keras.layers.Conv2DTranspose(32 * 4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32 * 4, up7)

    up8 = keras.layers.concatenate([keras.layers.Conv2DTranspose(32 * 2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32 * 2, up8)

    up9 = keras.layers.concatenate([keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9)
    if classes>1:
        d=keras.layers.Conv2D(classes,kernel_size=3,use_bias=False,activation=None)(mresblock9)
        d=keras.layers.Activation('softmax',name='d')(d)
    else:
        d=keras.layers.Conv2D(1,kernel_size=3,use_bias=False,activation=None)(mresblock9)
        d=keras.layers.Activation('sigmoid',name='d')(d)
    model = keras.models.Model(inputs=inputs, outputs=d)
    return model


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    input_shape = (256, 256, 3)
    model = MultiResUnet(input_shape=input_shape, classes=1)
    keras.utils.plot_model(model, 'MultiResUnet.png', show_shapes=True)
    print(model.summary())
