# -*- coding: utf-8 -*-
import tensorflow as tf
from .model import NetWork

# ['height', 'pressure', 'tempc', 'ua', 'va', 'wa', 'qv', 'qc', 'ex2', 'ssa2', 'g2', 'dens', 'no2', 'hcho', 'so2', 'o3', 'pm25', 'pm10']

class Aider(object):
    def __init__(self, batch_size, img_h, img_w, iptlev, outlev):
        
        self.env_t = tf.placeholder(tf.float32, [batch_size, 24])
        
        self.no2sum= tf.placeholder(tf.float32, [batch_size, img_h, img_w, iptlev])
        self.atm_P = tf.placeholder(tf.float32, [batch_size, img_h, img_w, iptlev])
        self.atm_T = tf.placeholder(tf.float32, [batch_size, img_h, img_w, iptlev])
        self.atm_u = tf.placeholder(tf.float32, [batch_size, img_h, img_w, iptlev])
        self.atm_v = tf.placeholder(tf.float32, [batch_size, img_h, img_w, iptlev])
        self.atm_w = tf.placeholder(tf.float32, [batch_size, img_h, img_w, iptlev])
        
        self.mask = tf.placeholder(tf.float32, [batch_size, img_h, img_w, iptlev])
        self.msk1 = tf.placeholder(tf.float32, [batch_size, img_h, img_w, 64])
        self.msk2 = tf.placeholder(tf.float32, [batch_size, img_h, img_w, 64])
        self.msk3 = tf.placeholder(tf.float32, [batch_size, 2])
        
        self.label = tf.placeholder(tf.float32, [batch_size, outlev])
        
        self.lr = tf.placeholder(tf.float32, shape=[])
        
        grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            # define a model
            self.list_input = [self.no2sum, self.atm_P, self.atm_T, 
                               self.atm_u, self.atm_v, self.atm_w, self.env_t]
            self.list_mask = [self.mask, self.msk1, self.msk2, self.msk3]
            self.pred, self.loss = NetWork(self.list_input, self.label, self.list_mask)
            
            # gradients
            all_params = tf.trainable_variables()
            grads.append(tf.gradients(self.loss, all_params))

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables, max_to_keep=100)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)
        self.sess.run(init)

    def train(self, list_input, label, list_msk, lr):
        feed_dict = {self.no2sum:list_input[0],
                     self.atm_P:list_input[1],
                     self.atm_T:list_input[2],
                     self.atm_u:list_input[3],
                     self.atm_v:list_input[4],
                     self.atm_w:list_input[5],
                     self.env_t:list_input[6],
                     self.mask:list_msk[0],
                     self.msk1:list_msk[1],
                     self.msk2:list_msk[2],
                     self.msk3:list_msk[3],
                     self.label:label,
                     self.lr:lr                 }
        loss, pred, _ = self.sess.run((self.loss, self.pred, self.train_op), feed_dict)
        return loss, pred
    
    def test(self, list_input, label, list_msk):
        feed_dict = {self.no2sum:list_input[0],
                     self.atm_P:list_input[1],
                     self.atm_T:list_input[2],
                     self.atm_u:list_input[3],
                     self.atm_v:list_input[4],
                     self.atm_w:list_input[5],
                     self.env_t:list_input[6],
                     self.mask:list_msk[0],
                     self.msk1:list_msk[1],
                     self.msk2:list_msk[2],
                     self.msk3:list_msk[3],
                     self.label:label           }
        loss, pred = self.sess.run((self.loss, self.pred), feed_dict)
        return loss, pred

    def save(self, filepath):
        self.saver.save(self.sess, filepath)
        print('saved to ' + filepath)
        
    def load(self, filepath):
        print('load model:', filepath)
        self.saver.restore(self.sess, filepath)


