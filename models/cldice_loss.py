from tensorflow.keras import layers as KL
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
#from .soft_skeleton import soft_skel
#https://github.com/jocpae/clDice
def soft_erode(img):
    """[This function performs soft-erosion operation on a float32 image]
    Args:
        img ([float32]): [image to be soft eroded]
    Returns:
        [float32]: [the eroded image]
    """
    if len(img.shape)==4:
        p1 = -KL.MaxPool2D(pool_size=(3, 1), strides=(1,  1), padding='same', data_format=None)(-img)
        p2 = -KL.MaxPool2D(pool_size=(1, 3), strides=(1,  1), padding='same', data_format=None)(-img)
        return tf.math.minimum(p1,p2)
    elif len(img.shape) == 5:
        p1 = -KL.MaxPool3D(pool_size=(3, 3, 1), strides=(1, 1, 1), padding='same', data_format=None)(-img)
        p2 = -KL.MaxPool3D(pool_size=(3, 1, 3), strides=(1, 1, 1), padding='same', data_format=None)(-img)
        p3 = -KL.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(-img)
        return tf.math.minimum(tf.math.minimum(p1, p2), p3)



def soft_dilate(img):
    """[This function performs soft-dilation operation on a float32 image]
    Args:
        img ([float32]): [image to be soft dialated]
    Returns:
        [float32]: [the dialated image]
    """
    if len(img.shape)==4:
        return  KL.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', data_format=None)(img)
    elif len(img.shape)==5:
        return KL.MaxPool3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(img)


def soft_open(img):
    """[This function performs soft-open operation on a float32 image]
    Args:
        img ([float32]): [image to be soft opened]
    Returns:
        [float32]: [image after soft-open]
    """
    img = soft_erode(img)
    img = soft_dilate(img)
    return img


def soft_skel(img, iters):
    """[summary]
    Args:
        img ([float32]): [description]
        iters ([int]): [description]
    Returns:
        [float32]: [description]
    """
    img1 = soft_open(img)
    skel = tf.nn.relu(img-img1)

    for j in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = tf.nn.relu(img-img1)
        intersect = tf.math.multiply(skel, delta)
        skel += tf.nn.relu(delta-intersect)
    return skel

def soft_clDice_loss(iter_ = 50):
    """[function to compute dice loss]
    Args:
        iter_ (int, optional): [skeletonization iteration]. Defaults to 50.
    """
    def loss(y_true, y_pred):
        """[function to compute dice loss]
        Args:
            y_true ([float32]): [ground truth image]
            y_pred ([float32]): [predicted image]
        Returns:
            [float32]: [loss value]
        """
        smooth = 1.
        skel_pred = soft_skel(y_pred, iter_)
        skel_true = soft_skel(y_true, iter_)
        pres = (K.sum(tf.math.multiply(skel_pred, y_true)[:,1:,:,:,:])+smooth)/(K.sum(skel_pred[:,1:,:,:,:])+smooth)
        rec = (K.sum(tf.math.multiply(skel_true, y_pred)[:,1:,:,:,:])+smooth)/(K.sum(skel_true[:,1:,:,:,:])+smooth)
        cl_dice = 1.- 2.0*(pres*rec)/(pres+rec)
        return cl_dice
    return loss


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]
    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]
    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = K.sum((y_true * y_pred)[:,1:,...])
    coeff = (2. *  intersection + smooth) / (K.sum(y_true[:,1:,...]) + K.sum(y_pred[:,1:,...]) + smooth)
    return (1. - coeff)


def soft_dice_cldice_loss(iters = 15, alpha=0.5):
    """[function to compute dice+cldice loss]
    Args:
        iters (int, optional): [skeletonization iteration]. Defaults to 15.
        alpha (float, optional): [weight for the cldice component]. Defaults to 0.5.
    """
    def loss(y_true, y_pred):
        """[summary]
        Args:
            y_true ([float32]): [ground truth image]
            y_pred ([float32]): [predicted image]
        Returns:
            [float32]: [loss value]
        """
        smooth = 1.
        skel_pred = soft_skel(y_pred, iters)
        skel_true = soft_skel(y_true, iters)
        #pres = (K.sum(tf.math.multiply(skel_pred, y_true)[:,1:,:,:,:])+smooth)/(K.sum(skel_pred[:,1:,:,:,:])+smooth)
        #rec = (K.sum(tf.math.multiply(skel_true, y_pred)[:,1:,:,:,:])+smooth)/(K.sum(skel_true[:,1:,:,:,:])+smooth)
        pres = (K.sum(tf.math.multiply(skel_pred, y_true)[:, 1:, ...]) + smooth) / ( K.sum(skel_pred[:, 1:, ...]) + smooth)
        rec = (K.sum(tf.math.multiply(skel_true, y_pred)[:, 1:, ...]) + smooth) / (K.sum(skel_true[:, 1:, ...]) + smooth)
        cl_dice = 1.- 2.0*(pres*rec)/(pres+rec)
        dice = soft_dice(y_true, y_pred)
        return (1.0-alpha)*dice+alpha*cl_dice
    return loss
