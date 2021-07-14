import cv2
import numpy as np
from random import choice


class ImageRead(object):
    def __call__(self, *inputs):
        imgs = []
        for x in inputs:
            imgs.append( cv2.imread(x) )
        return imgs
    
    
class ToRGB(object):
    def __call__(self, *inputs):
        return  [img.convert('RGB') for img in inputs]

class resize(object):
    def __init__(self, size):
        self.size = size
                 
    def __call__(self, *inputs):
        return [img.resize(self.size) for img in inputs]



class img2patch(object):
    def __init__(self, h, w, patchsize):
        self.size = h
        
    def __call__(self, *inputs):
        return [img.resize(self.size) for img in inputs]


class RandomFlip(object):
        
    def __call__(self, *inputs):
        if choice(range(2)):
            return [img.resize(self.size) for img in inputs]

