from tensorflow import keras
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
#from  medpy.metric import dc
#https://stats.stackexchange.com/questions/195006/is-the-dice-coefficient-the-same-as-accuracy
def IOU(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred = keras.backend.cast(y_pred, 'float32')
    y_pred_f = keras.backend.cast(keras.backend.greater(keras.backend.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    union = keras.backend.sum(y_true) + keras.backend.sum(y_pred) - intersection
    return intersection / (union+keras.backend.epsilon())

def DICE(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred = keras.backend.cast(y_pred, 'float32')
    y_pred_f = keras.backend.cast(keras.backend.greater(keras.backend.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * keras.backend.sum(intersection) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f)+K.epsilon())
    return score


def dc(result, reference):
    r""" Dice coefficient
    Computes the Dice coefficient (also known as Sorensen index) between the binary objects in two images.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    intersection = np.count_nonzero(result & reference)
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    return dc

def dice_np(y_true, y_pred,a=0.5):
    y_true_f=y_true.flatten()
    y_pred_f=y_pred.flatten()
    y_pred_f=np.greater(y_pred_f,a)
    score=dc(y_true_f,y_pred_f)
    #intersection = y_true_f * y_pred_f
    #score=2.*intersection.sum()/(y_pred_f.sum()+y_true_f.sum()+0.00000001)
    return score

def find_best_dice(y_true, y_pred,step=0.001):
    bst=0.
    alpha=0.
    for a in np.arange(0, 1., step):
        score=dice_np(y_true, y_pred, a=a)
        if score>bst:
            bst=score
            alpha=a
            print(a,score)
    return bst,alpha




def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def focal_loss(y_true, y_pred):
    gamma = 2
    alpha = 0.25
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
    return K.mean(focal_loss)


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * keras.backend.sum(intersection) + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)+focal_loss(y_true, y_pred)


def find_alpha(y_true,y_pred,step=0.01):
    n=len(y_true)
    out=[]
    alpha=best_f1=0.
    for a in np.arange(0,1.,step):
        y_hat=np.zeros((n))
        y_hat[y_pred>=a]=1
        #cf=metrics.confusion_matrix(y_true,y_hat)
        recall=metrics.recall_score(y_true,y_hat)
        precision=metrics.precision_score(y_true,y_hat)
        f1=metrics.f1_score(y_true,y_hat)
        out.append((a,f1,recall,precision))
        if f1> best_f1:
            best_f1=f1
            alpha=a
            print('alpha:%.6f,F1:%.6f,Recall:%.6f,Precision:%.6f'%(alpha,f1,recall,precision) )
    out=pd.DataFrame(out,columns=['alpha','F1','Recall','Precision'])
    print(out[out.alpha==alpha])
    print(out[out.Precision == out.Precision.max()])
    return out




