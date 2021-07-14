#! -*- coding:utf-8 -*-
from keras.models import Model
from keras.layers.noise import GaussianNoise
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.normalization import BatchNormalization
K.set_image_dim_ordering('tf')

def get_Cnet(slice_h, slice_w, img_ch=5, netname='Cnet'):
    
    fundus = Input((slice_h, slice_w, img_ch))
    vessel = Input((slice_h, slice_w, 1))
    vein   = Input((slice_h, slice_w, 1))
    
    Cinput = Dropout(0.1)(fundus)
    Cinput = Concatenate(axis=3)([Cinput, vessel, vein])
    
    # h, w
    Ce1_conv = Conv2D(32, 3, padding ='same', activation='relu')(Cinput)
    Ce1_conv = BatchNormalization(scale=False, axis=3 )(Ce1_conv)
    Ce1_conv = Dropout(0.2)(Ce1_conv)
    
    Ce1_conv = Conv2D(32, 3, padding ='same', activation='relu')(Ce1_conv)
    Ce1_conv = BatchNormalization(scale=False, axis=3 )(Ce1_conv)
    Ce1_pool = MaxPooling2D(pool_size=(2, 2))(Ce1_conv)
    
    
    # h/2, w/2
    Ce2_conv = Conv2D(64, 3, padding ='same', activation='relu')(Ce1_pool)
    Ce2_conv = BatchNormalization(scale=False, axis=3 )(Ce2_conv)
    Ce2_conv = Dropout(0.2)(Ce2_conv)
    
    Ce2_conv = Conv2D(64, 3, padding ='same', activation='relu')(Ce2_conv)
    Ce2_conv = BatchNormalization(scale=False, axis=3 )(Ce2_conv)
    Ce2_pool = MaxPooling2D(pool_size=(2, 2))(Ce2_conv)
    
    
    # h/4, w/4
    Ce3_conv = Conv2D(128, 3, padding ='same', activation='relu')(Ce2_pool)
    Ce3_conv = BatchNormalization(scale=False, axis=3 )(Ce3_conv)
    Ce3_conv = Dropout(0.2)(Ce3_conv)
     
    Ce3_conv = Conv2D(128, 3, padding ='same', activation='relu')(Ce3_conv)
    Ce3_conv = BatchNormalization(scale=False, axis=3 )(Ce3_conv)
    Ce3_pool = MaxPooling2D(pool_size=(2, 2))(Ce3_conv)
    
    
    # h/8, w/8
    Ce4_conv = Conv2D(256, 3, padding ='same', activation='relu')(Ce3_pool)
    Ce4_conv = BatchNormalization(scale=False, axis=3 )(Ce4_conv)
    Ce4_conv = Dropout(0.2)(Ce4_conv)
    
    Ce4_conv = Conv2D(256, 3, padding ='same', activation='relu')(Ce4_conv)
    Ce4_conv = BatchNormalization(scale=False, axis=3 )(Ce4_conv)
    
    
    Cre4_conv = Conv2D(128, 3, padding ='same', activation='relu')(Ce4_conv)
    Cre4_conv = BatchNormalization(scale=False, axis=3 )(Cre4_conv)
    Cre4_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='Cre4_out')(Cre4_conv)

    
    # ------------------  A_decode   -------------------
    # h/4, w/4
    Cre3_conv = Conv2D(128, 3, padding ='same', activation='relu')(Ce3_conv)
    Cre3_conv = BatchNormalization(scale=False, axis=3 )(Cre3_conv)
    Cre3_conv = Dropout(0.2)(Cre3_conv)
    
    Cre3_conv = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(Cre4_conv), Cre3_conv])
    Cre3_conv = Conv2D(64, 3, padding ='same', activation='relu')(Cre3_conv)
    Cre3_conv = BatchNormalization(scale=False, axis=3 )(Cre3_conv)
    Cre3_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='Cre3_out')(Cre3_conv)
    
    # h/2, w/2
    Cre2_conv = Conv2D(64, 3, padding ='same', activation='relu')(Ce2_conv)
    Cre2_conv = BatchNormalization(scale=False, axis=3 )(Cre2_conv)
    Cre2_conv = Dropout(0.2)(Cre2_conv)
    
    Cre2_conv = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(Cre3_conv), Cre2_conv])
    Cre2_conv = Conv2D(32, 3, padding ='same', activation='relu')(Cre2_conv)
    Cre2_conv = BatchNormalization(scale=False, axis=3 )(Cre2_conv)
    Cre2_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='Cre2_out')(Cre2_conv)
    
    # h, w
    Cre1_conv = Conv2D(32, 3, padding ='same', activation='relu')(Ce1_conv)
    Cre1_conv = BatchNormalization(scale=False, axis=3 )(Cre1_conv)
    Cre1_conv = Dropout(0.2)(Cre1_conv)
    
    Cre1_conv = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(Cre2_conv), Cre1_conv])
    Cre1_conv = Conv2D(16, 3, padding ='same', activation='relu')(Cre1_conv)
    Cre1_conv = BatchNormalization(scale=False, axis=3 )(Cre1_conv)
    Cre1_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='Cre1_out')(Cre1_conv)
    
    model = Model(input=[fundus, vessel, vein], output=[Cre1_out,Cre2_out,Cre3_out,Cre4_out], name=netname)
    
    return model