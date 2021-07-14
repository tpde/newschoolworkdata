#! -*- coding:utf-8 -*-
from keras.models import Model
from keras.layers.noise import GaussianNoise
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.normalization import BatchNormalization
K.set_image_dim_ordering('tf')

def get_Anet(slice_h, slice_w):
    
    A_fundus = Input((slice_h, slice_w, 5))
    
#     A_fundus = Concatenate(axis=3)([fundus_rgb, fundus_610, fundus_570])
    Ae1_conv = Dropout(0.1)(A_fundus)
    
    # h, w
    Ae1_conv = Conv2D(32, 3, padding ='same', activation='relu')(Ae1_conv)
    Ae1_conv = BatchNormalization(scale=False, axis=3 )(Ae1_conv)
    Ae1_conv = Dropout(0.2)(Ae1_conv)
    
    Ae1_conv = Conv2D(32, 3, padding ='same', activation='relu')(Ae1_conv)
    Ae1_conv = BatchNormalization(scale=False, axis=3 )(Ae1_conv)
    Ae1_pool = MaxPooling2D(pool_size=(2, 2))(Ae1_conv)
    
    
    # h/2, w/2
    Ae2_conv = Conv2D(64, 3, padding ='same', activation='relu')(Ae1_pool)
    Ae2_conv = BatchNormalization(scale=False, axis=3 )(Ae2_conv)
    Ae2_conv = Dropout(0.2)(Ae2_conv)
    
    Ae2_conv = Conv2D(64, 3, padding ='same', activation='relu')(Ae2_conv)
    Ae2_conv = BatchNormalization(scale=False, axis=3 )(Ae2_conv)
    Ae2_pool = MaxPooling2D(pool_size=(2, 2))(Ae2_conv)
    
    
    # h/4, w/4
    Ae3_conv = Conv2D(128, 3, padding ='same', activation='relu')(Ae2_pool)
    Ae3_conv = BatchNormalization(scale=False, axis=3 )(Ae3_conv)
    Ae3_conv = Dropout(0.2)(Ae3_conv)
     
    Ae3_conv = Conv2D(128, 3, padding ='same', activation='relu')(Ae3_conv)
    Ae3_conv = BatchNormalization(scale=False, axis=3 )(Ae3_conv)
    Ae3_pool = MaxPooling2D(pool_size=(2, 2))(Ae3_conv)
    
    
    # h/8, w/8
    Ae4_conv = Conv2D(256, 3, padding ='same', activation='relu')(Ae3_pool)
    Ae4_conv = BatchNormalization(scale=False, axis=3 )(Ae4_conv)
    Ae4_conv = Dropout(0.2)(Ae4_conv)
    
    Ae4_conv = Conv2D(256, 3, padding ='same', activation='relu')(Ae4_conv)
    Ae4_conv = BatchNormalization(scale=False, axis=3 )(Ae4_conv)
    
    
    Are4_conv = Conv2D(128, 3, padding ='same', activation='relu')(Ae4_conv)
    Are4_conv = BatchNormalization(scale=False, axis=3 )(Are4_conv)
    Are4_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='Are4_out')(Are4_conv)

    
    # ------------------  A_decode   -------------------
    # h/4, w/4
    Are3_conv = Conv2D(128, 3, padding ='same', activation='relu')(Ae3_conv)
    Are3_conv = BatchNormalization(scale=False, axis=3 )(Are3_conv)
    Are3_conv = Dropout(0.2)(Are3_conv)
    
    Are3_conv = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(Are4_conv), Are3_conv])
    Are3_conv = Conv2D(64, 3, padding ='same', activation='relu')(Are3_conv)
    Are3_conv = BatchNormalization(scale=False, axis=3 )(Are3_conv)
    Are3_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='Are3_out')(Are3_conv)
    
    # h/2, w/2
    Are2_conv = Conv2D(64, 3, padding ='same', activation='relu')(Ae2_conv)
    Are2_conv = BatchNormalization(scale=False, axis=3 )(Are2_conv)
    Are2_conv = Dropout(0.2)(Are2_conv)
    Are2_conv = Conv2D(64, 3, padding ='same', activation='relu')(Are2_conv)
    Are2_conv = BatchNormalization(scale=False, axis=3 )(Are2_conv)
    Are2_conv = Dropout(0.2)(Are2_conv)
    
    Are2_conv = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(Are3_conv), Are2_conv])
    Are2_conv = Conv2D(32, 3, padding ='same', activation='relu')(Are2_conv)
    Are2_conv = BatchNormalization(scale=False, axis=3 )(Are2_conv)
    Are2_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='Are2_out')(Are2_conv)
    
    # h, w
    Are1_conv = Conv2D(32, 3, padding ='same', activation='relu')(Ae1_conv)
    Are1_conv = BatchNormalization(scale=False, axis=3 )(Are1_conv)
    Are1_conv = Dropout(0.2)(Are1_conv)
    Are1_conv = Conv2D(32, 3, padding ='same', activation='relu')(Are1_conv)
    Are1_conv = BatchNormalization(scale=False, axis=3 )(Are1_conv)
    Are1_conv = Dropout(0.2)(Are1_conv)
    
    Are1_conv = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(Are2_conv), Are1_conv])
    Are1_conv = Conv2D(16, 3, padding ='same', activation='relu')(Are1_conv)
    Are1_conv = BatchNormalization(scale=False, axis=3 )(Are1_conv)
    Are1_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='Are1_out')(Are1_conv)
    
    model = Model(input=A_fundus, output=[Are1_out,Are2_out,Are3_out,Are4_out], name='Anet')
    return model