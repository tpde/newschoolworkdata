#! -*- coding:utf-8 -*-
from keras.models import Model
from keras.layers.noise import GaussianNoise
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.normalization import BatchNormalization
K.set_image_dim_ordering('tf')

def get_Bnet(slice_h, slice_w, img_ch=5, netname='Bnet'):
    
    fundus = Input((slice_h, slice_w, img_ch))
    vessel = Input((slice_h, slice_w, 1))
    
    Binput = Dropout(0.1)(fundus)
    Binput = Concatenate(axis=3)([Binput, vessel])
    
    
    # h, w
    Be1_conv = Conv2D(32, 3, padding ='same', activation='relu')(Binput)
    Be1_conv = BatchNormalization(scale=False, axis=3 )(Be1_conv)
    Be1_conv = Dropout(0.2)(Be1_conv)
    
    Be1_conv = Conv2D(32, 3, padding ='same', activation='relu')(Be1_conv)
    Be1_conv = BatchNormalization(scale=False, axis=3 )(Be1_conv)
    Be1_pool = MaxPooling2D(pool_size=(2, 2))(Be1_conv)
    
    
    # h/2, w/2
    Be2_conv = Conv2D(64, 3, padding ='same', activation='relu')(Be1_pool)
    Be2_conv = BatchNormalization(scale=False, axis=3 )(Be2_conv)
    Be2_conv = Dropout(0.2)(Be2_conv)
    
    Be2_conv = Conv2D(64, 3, padding ='same', activation='relu')(Be2_conv)
    Be2_conv = BatchNormalization(scale=False, axis=3 )(Be2_conv)
    Be2_pool = MaxPooling2D(pool_size=(2, 2))(Be2_conv)
    
    
    # h/4, w/4
    Be3_conv = Conv2D(128, 3, padding ='same', activation='relu')(Be2_pool)
    Be3_conv = BatchNormalization(scale=False, axis=3 )(Be3_conv)
    Be3_conv = Dropout(0.2)(Be3_conv)
     
    Be3_conv = Conv2D(128, 3, padding ='same', activation='relu')(Be3_conv)
    Be3_conv = BatchNormalization(scale=False, axis=3 )(Be3_conv)
    Be3_pool = MaxPooling2D(pool_size=(2, 2))(Be3_conv)
    
    
    # h/8, w/8
    Be4_conv = Conv2D(256, 3, padding ='same', activation='relu')(Be3_pool)
    Be4_conv = BatchNormalization(scale=False, axis=3 )(Be4_conv)
    Be4_conv = Dropout(0.2)(Be4_conv)
    
    Be4_conv = Conv2D(256, 3, padding ='same', activation='relu')(Be4_conv)
    Be4_conv = BatchNormalization(scale=False, axis=3 )(Be4_conv)
    
    
    Bre4_conv = Conv2D(128, 3, padding ='same', activation='relu')(Be4_conv)
    Bre4_conv = BatchNormalization(scale=False, axis=3 )(Bre4_conv)
    Bre4_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='Bre4_out')(Bre4_conv)

    
    # ------------------  A_decode   -------------------
    # h/4, w/4
    Bre3_conv = Conv2D(128, 3, padding ='same', activation='relu')(Be3_conv)
    Bre3_conv = BatchNormalization(scale=False, axis=3 )(Bre3_conv)
    Bre3_conv = Dropout(0.2)(Bre3_conv)
    
    Bre3_conv = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(Bre4_conv), Bre3_conv])
    Bre3_conv = Conv2D(64, 3, padding ='same', activation='relu')(Bre3_conv)
    Bre3_conv = BatchNormalization(scale=False, axis=3 )(Bre3_conv)
    Bre3_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='Bre3_out')(Bre3_conv)
    
    # h/2, w/2
    Bre2_conv = Conv2D(64, 3, padding ='same', activation='relu')(Be2_conv)
    Bre2_conv = BatchNormalization(scale=False, axis=3 )(Bre2_conv)
    Bre2_conv = Dropout(0.2)(Bre2_conv)
    Bre2_conv = Conv2D(64, 3, padding ='same', activation='relu')(Bre2_conv)
    Bre2_conv = BatchNormalization(scale=False, axis=3 )(Bre2_conv)
    Bre2_conv = Dropout(0.2)(Bre2_conv)
    
    Bre2_conv = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(Bre3_conv), Bre2_conv])
    Bre2_conv = Conv2D(32, 3, padding ='same', activation='relu')(Bre2_conv)
    Bre2_conv = BatchNormalization(scale=False, axis=3 )(Bre2_conv)
    Bre2_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='Bre2_out')(Bre2_conv)
    
    # h, w
    Bre1_conv = Conv2D(32, 3, padding ='same', activation='relu')(Be1_conv)
    Bre1_conv = BatchNormalization(scale=False, axis=3 )(Bre1_conv)
    Bre1_conv = Dropout(0.2)(Bre1_conv)
    Bre1_conv = Conv2D(32, 3, padding ='same', activation='relu')(Bre1_conv)
    Bre1_conv = BatchNormalization(scale=False, axis=3 )(Bre1_conv)
    Bre1_conv = Dropout(0.2)(Bre1_conv)
    
    Bre1_conv = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(Bre2_conv), Bre1_conv])
    Bre1_conv = Conv2D(16, 3, padding ='same', activation='relu')(Bre1_conv)
    Bre1_conv = BatchNormalization(scale=False, axis=3 )(Bre1_conv)
    Bre1_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='Bre1_out')(Bre1_conv)
    
    model = Model(input=[fundus,vessel], output=[Bre1_out,Bre2_out,Bre3_out,Bre4_out], name=netname)
    
    return model