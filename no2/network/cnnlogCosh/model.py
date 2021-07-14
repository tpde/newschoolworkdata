import tensorflow as tsf
from tensorflow import layers
from .ops import resblock, flatten, circleAttention1, circleAttention2, levelAttention


def NetWork(list_input, label, list_mask):
    
    outdim = label.get_shape().as_list()[-1]
    
    with tsf.variable_scope('NetWork'):
        list_x = []
        for i,x in enumerate( list_input[:-1] ):
#             x = levelAttention(x, list_mask[0])
            x = layers.conv2d(x, 64, 1, strides=(1, 1), padding='same', activation='relu')
#             x = circleAttention1(x, list_mask[1:])
#             x = circleAttention2(x, list_mask[1:])
            # for j,flt in enumerate( list_flt1 ):
            #     x = resblock('ecode_ipt%d_l%d'%(i,j), x, flt, ksize, is_training=is_training)
            x = layers.max_pooling2d(x, (2,2), strides=2, padding='same')
            x = layers.conv2d(x, 64, 3, strides=(1, 1), padding='same', activation='relu')
#             x = layers.conv2d(x, 16, 1, strides=(1, 1), padding='same', activation='relu')
            list_x.append( x )
            
        x = tsf.concat( list_x, 3)
        x = tsf.reduce_mean(x, axis=(1,2), keepdims=True)
        x = flatten(x)
        x = tsf.concat( [x, list_input[-1]], 1)
        x = layers.dense(x, int(x.get_shape()[-1]//4), activation='relu')
        # x = layers.dense(x, int(x.get_shape()[-1]//2), activation='relu')
        pred = layers.dense(x, outdim)

    # loss = tsf.nn.l2_loss(arrpred - arrlabel)
    loss = tsf.keras.losses.logcosh(label, pred)
    return [pred, loss]


