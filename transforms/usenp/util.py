import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import numbers
import collections
import sys, random
from scipy.ndimage import rotate
from random import choice


if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class ToNumpy(object):
    def __call__(self, *inputs):
        return [np.array(img)[...,np.newaxis] if img.mode=='L' else np.array(img) for img in inputs]

class StandardScaler(object):
    def __call__(self, *inputs):
        return [(X-X.mean())/(X.std()) if np.any(X!=0) else X for X in inputs]

class Zero_center(object):
    def __init__(self, mean=[103.939, 116.779, 123.68], is_cv2read=False):
        self.mean = mean
        self.is_cv2read = is_cv2read
        
        
    def __call__(self, *inputs):
        result_list = []
        for x in inputs:
            if self.is_cv2read:
                x = x[..., ::-1].astype(np.float64)
            if isinstance(self.mean, (tuple,list)):
                x[..., 0] -= self.mean[0]
                x[..., 1] -= self.mean[1]
                x[..., 2] -= self.mean[2]
            else:
                x -= self.mean
            result_list.append(x)
        return result_list


class TileChannel(object):
    def __init__(self, axis, times):
        self.axis=axis
        self.times=times
        
    def __call__(self, *inputs):
        return [np.concatenate([arr for i in range(self.times)], self.axis) for arr in inputs]



class MinMaxScaler(object):

    def __init__(self, feature_range=(0,1)):
        """
        Scale data between an arbitrary range
        """
        self.feature_range = feature_range
    
    def __call__(self, *inputs):
        list_result = []
        for X in inputs:
            if np.any(X!=0):
                min_x = X.min()
                max_x = X.max()
                scale_ = (self.feature_range[1] - self.feature_range[0]) / (max_x - min_x)
                min_ = self.feature_range[0] - min_x * scale_
                X *= scale_
                X += min_
            list_result.append(X)
        return list_result


class ExpandDims(object):

    def __init__(self, axis=-1):
        self.axis = axis
    
    def __call__(self, *inputs):
        return [np.expand_dims(X, axis=self.axis) for X in inputs]



class TypeCast(object):
    def __init__(self, dtype):
        if not isinstance(dtype, (tuple,list)):
            self.dtype = (dtype,dtype)
    
    def __call__(self, *inputs):
        return [X.astype(self.dtype[0]) for X in inputs]

        
class RandomCrop(object):
    def __init__(self, size, padding=0):
        self.size = size
        self.pad = padding
 
    def __call__(self, *inputs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        """
        x = random.randint(0-self.pad, inputs[0].size[0]+self.pad-self.size[0])
        y = random.randint(0-self.pad, inputs[0].size[1]+self.pad-self.size[1])
        return [img.crop(box=(x,y,self.size[0]+x, self.size[1]+y)) for img in inputs]

            
            
class BinaryMask(object):
    def __init__(self, thresh):
        if thresh >= 1:
            raise ValueError('cutoff must be less than 1')
        self.thresh = thresh
    
    def __call__(self, *inputs):
        return [(X>self.thresh).astype(np.uint8) for X in inputs]

class RandomFlip(object):
    def __call__(self, *inputs):
        if choice(range(2)):
            inputs = [input[::-1] for input in inputs]
        if choice(range(2)):
            inputs = [input[:,::-1] for input in inputs]
        return inputs
    
class RandomRotate(object):
    def __init__(self, list_isbound):
        self.list_isbound = list_isbound
        
    def __call__(self, *inputs):
        angle = choice(range(360))
        inputs = [rotate(input, angle, axes=(0, 1), reshape=False) for input in inputs]
        return [np.round(input) if self.list_isbound[i] else input for i,input in enumerate(inputs)]


class Padding(object):
    """        
    padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
        - constant: pads with a constant value, this value is specified with fill
        - edge: pads with the last value on the edge of the image
        - reflect: pads with reflection of image (without repeating the last value on the edge)
                   padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                   will result in [3, 2, 1, 2, 3, 4, 3, 2]
        - symmetric: pads with reflection of image (repeating the last value on the edge)
                     padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                     will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def __init__(self, padding):
        self.padding = padding
    
    def __call__(self, *inputs):
        return [np.pad(input, self.padding, 'constant') for input in inputs]

