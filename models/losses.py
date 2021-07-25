import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
from cldice_loss import soft_clDice_loss,soft_dice_cldice_loss

def getLoss(name='bce_dice_focal'):
    if name=='bce_dice_focal':
        return bce_dice_focal_loss
    elif name=='bce_focal':
        return bce_focal_loss
    elif name=='bce_tversky':
        return bce_tversky_loss
    elif name=='focal':
        return focal_loss
    elif name=='bce_dice':
        return  bce_dice_loss
    elif name=='bce':
        return keras.losses.binary_crossentropy
    elif name=='dice':
        return dice_loss
    elif name=='cldice':
        return soft_dice_cldice_loss()
    else:
        return bce_dice_focal_loss
    
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
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def tversky(y_true, y_pred):
    smooth = 1.
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def generalized_dice_coeff(y_true, y_pred):
    #Ncl = y_pred.shape[-1]
    #w = K.zeros(shape=(Ncl,))
    y_true= K.flatten(y_true)
    y_pred= K.flatten(y_pred)

    w = K.sum(y_true)
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator)
    numerator = K.sum(numerator)
    denominator = y_true+y_pred
    denominator = w*K.sum(denominator)
    denominator = K.sum(denominator)
    gen_dice_coef = (2*numerator+1.)/(denominator+1.)
    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def bce_dice_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_dice_focal_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)+focal_loss(y_true, y_pred)

def bce_tversky_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) + tversky_loss(y_true, y_pred)

def bce_focal_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) + focal_loss(y_true, y_pred)
