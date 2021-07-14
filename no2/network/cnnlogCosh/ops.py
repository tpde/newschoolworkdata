import numpy as np
import tensorflow as tsf
from tensorflow import layers
import tensorflow.contrib as tf_contrib


def relu(x):
    return tsf.nn.relu(x)

def flatten(x) :
    return tsf.layers.flatten(x)
       
def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

def resblock(scope, x, flt, ksize, is_training=True) :
    with tsf.variable_scope(scope):
        res = layers.conv2d(x, flt, ksize, strides=(1, 1), padding='same')
        res = batch_norm(res, is_training, scope='batch_norm_1')
        res = relu(res)
        res = layers.conv2d(res, flt, ksize, strides=(1, 1), padding='same')
        if x.get_shape()[-1]!= flt:
            x = layers.conv2d(x, flt, 1, strides=(1, 1), padding='same')
            
        x = batch_norm(x+res, is_training, scope='batch_norm_0')
        x = relu(x)
        return x


def circleAttention1(x, list_msk):
    b, h, w, c = x.get_shape()
    atten = layers.dense(list_msk[-1], 2, 'relu')
    y = 0
    for k in range( min(h//2,w//2) ):
        att = tsf.expand_dims(tsf.expand_dims(atten[:,k:k+1], axis=-1),axis=-1)
        y += att * x * list_msk[k]
    return y


def circleAttention2(x, list_msk):
    b, h, w, c = x.get_shape()
    list_att = []
    for k in range( min(h//2,w//2) ):
        att = tsf.reduce_sum(x*list_msk[k], axis=(1,2))/tsf.reduce_sum(list_msk[k], axis=(1,2))
        list_att.append( tsf.expand_dims(tsf.expand_dims(att,axis=1),axis=1) )
        
    atten = tsf.concat( list_att, 2 )
    atten = layers.conv2d(atten, c, (1,2), strides=(1,1), padding='same')
    y = 0
    for k in range( min(h//2,w//2) ):
        y += atten[:,:,k:k+1] * x * list_msk[k]
    return y

# def circleAttention(x, list_msk):
#     b, h, w, c = x.get_shape()
#     atten = layers.conv2d(list_msk[0], c, 1, strides=(1, 1), padding='same')
#     y = 0
#     for k in range( min(h//2,w//2) ):
#         msk = list_msk[k+1]
#         att = tsf.reduce_sum(atten*msk, axis=(1,2)) / tsf.reduce_sum(msk, axis=(1,2))
#         y += tsf.expand_dims(tsf.expand_dims(att,axis=1),axis=1) *x*msk
#     return y

def levelAttention(x, mask):
    b, h, w, c = x.get_shape()
    
    atten = layers.conv2d(mask, c, 1, strides=(1, 1), padding='same')
    atten = layers.average_pooling2d(atten, pool_size=(h, w), strides=(1,1), padding='valid')
    y = atten*x
    return y


# mask = np.ones( x.get_shape() )
# mask = tsf.convert_to_tensor( mask )
# mask = tsf.cast(mask, tsf.float32)
# 
# msk = np.zeros( (b, h, w, c) )
# msk[:,   k:h-k,     k:w-k,  :] = 1.
# msk[:, k+1:h-k-1, k+1:w-k-1,:] = 0.
# msk = tsf.convert_to_tensor( msk )
# msk = tsf.cast(msk, tsf.float32)