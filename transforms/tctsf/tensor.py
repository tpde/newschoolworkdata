import os, math, random
import numpy as np
from PIL import Image
import torch as tc
from torch.autograd import Variable


class ToTensor(object):
    def __init__(self, div=1):
        self.div = div
        
    def __call__(self, *inputs):
        return [tc.from_numpy(input/self.div) for input in inputs]

class AddChannel(object):
    def __init__(self, axis=0):
        self.axis = axis
    
    def __call__(self, *inputs):
        return [input.unsqueeze(self.axis) if input.dim()<3 else input for input in inputs]


class ToCuda(object):
    def __init__(self, device):
        self.device = device
    def __call__(self, *inputs):
        return [input.cuda() for input in inputs]


class ToNpy(object):
    def __init__(self, save_dir, list_name=None):
        self.save_dir = save_dir
        self.list_name = list_name
        self.idx = 0
        
    def __call__(self, *inputs):
        for input in inputs:
            fpath = os.path.join(self.save_dir, '%d'%(self.idx) if self.list_name is None else self.list_name[self.idx])
            np.save(fpath, input.numpy())
            self.idx += 1
    
class ToPil(object):
    def __init__(self, save_dir, list_name=None):
        self.save_dir = save_dir
        self.list_name = list_name
        self.idx = 0

    def __call__(self, *inputs):
        for input in inputs:
            fpath = os.path.join(self.save_dir, '%d.png'%(self.idx) if self.list_name is None else self.list_name[self.idx])
            input = input.squeeze().numpy()
            input = input*255 if np.max(input)<=1 else input
            img = Image.fromarray(input.astype(np.uint8))
            img.save(fpath)
            self.idx += 1

class ChannelLast(object):
    def __call__(self, *inputs):
        ndim = inputs[0].dim()
        if inputs[0].size(0) > inputs[0].size(-1):
            return inputs
        else:
            plist = list(range(1,ndim))+[0]
            return [input.permute(*plist) for input in inputs]

class ChannelFirst(object):
    def __call__(self, *inputs):
        ndim = inputs[0].dim()
        if inputs[0].size(0) < inputs[0].size(-1):
            return inputs
        else:
            plist = [ndim-1] + list(range(0,ndim-1))
            return [input.permute(*plist) for input in inputs]

class TypeCast(object):
    def __init__(self, dtypes=['float']):
        """
        # Cast a torch.Tensor to a different type
        dtype : string or torch.*Tensor literal or list of such
            data type to which input(s) will be cast.
            If list, it should be the same length as inputs.
        """
        dict_dtype = {  'byte' : tc.ByteTensor,
                        'double': tc.DoubleTensor,
                        'float': tc.FloatTensor,
                        'int' : tc.IntTensor,
                        'long': tc.LongTensor,
                        'short':tc.ShortTensor }
        self.dtypes = [dict_dtype[dtype] if isinstance(dtype,str) else dtype for dtype in dtypes]

    def __call__(self, *inputs):
        assert( len(self.dtypes)==len(inputs) )
        return [input.type(self.dtypes[idx]) for idx,input in enumerate(inputs)]

class Transpose(object):

    def __init__(self, dim1, dim2):
        """
        Swaps two dimensions of a tensor

        Arguments
        ---------
        dim1 : integer
            first dim to switch
        dim2 : integer
            second dim to switch
        """
        self.dim1 = dim1
        self.dim2 = dim2

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = tc.transpose(_input, self.dim1, self.dim2)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


class RangeNormalize(object):
    """
    Given min_val: (R, G, B) and max_val: (R,G,B),
    will normalize each channel of the tc.*Tensor to
    the provided min and max values.

    Works by calculating :
        a = (max'-min')/(max-min)
        b = max' - a * max
        new_value = a * value + b
    where min' & max' are given values, 
    and min & max are observed min/max for each channel
    
    Arguments
    ---------
    min_range : float or integer
        Min value to which tensors will be normalized
    max_range : float or integer
        Max value to which tensors will be normalized
    fixed_min : float or integer
        Give this value if every sample has the same min (max) and 
        you know for sure what it is. For instance, if you
        have an image then you know the min value will be 0 and the
        max value will be 255. Otherwise, the min/max value will be
        calculated for each individual sample and this will decrease
        speed. Dont use this if each sample has a different min/max.
    fixed_max :float or integer
        See above

    Example:
        >>> x = tc.rand(3,5,5)
        >>> rn = RangeNormalize((0,0,10),(1,1,11))
        >>> x_norm = rn(x)

    Also works with just one value for min/max:
        >>> x = tc.rand(3,5,5)
        >>> rn = RangeNormalize(0,1)
        >>> x_norm = rn(x)
    """
    def __init__(self, 
                 min_val, 
                 max_val):
        """
        Normalize a tensor between a min and max value

        Arguments
        ---------
        min_val : float
            lower bound of normalized tensor
        max_val : float
            upper bound of normalized tensor
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _min_val = _input.min()
            _max_val = _input.max()
            a = (self.max_val - self.min_val) / (_max_val - _min_val)
            b = self.max_val- a * _max_val
            _input = _input.mul(a).add(b)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


class StdNormalize(object):
    """
    Normalize torch tensor to have zero mean and unit std deviation
    """
    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.sub(_input.mean()).div(_input.std())
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


class Slice2D(object):

    def __init__(self, axis=0, reject_zeros=False):
        """
        Take a random 2D slice from a 3D image along 
        a given axis. This image should not have a 4th channel dim.

        Arguments
        ---------
        axis : integer in {0, 1, 2}
            the axis on which to take slices

        reject_zeros : boolean
            whether to reject slices that are all zeros
        """
        self.axis = axis
        self.reject_zeros = reject_zeros

    def __call__(self, x, y=None):
        while True:
            keep_slice  = random.randint(0,x.size(self.axis)-1)
            if self.axis == 0:
                slice_x = x[keep_slice,:,:]
                if y is not None:
                    slice_y = y[keep_slice,:,:]
            elif self.axis == 1:
                slice_x = x[:,keep_slice,:]
                if y is not None:
                    slice_y = y[:,keep_slice,:]
            elif self.axis == 2:
                slice_x = x[:,:,keep_slice]
                if y is not None:
                    slice_y = y[:,:,keep_slice]

            if not self.reject_zeros:
                break
            else:
                if y is not None and tc.sum(slice_y) > 0:
                        break
                elif tc.sum(slice_x) > 0:
                        break
        if y is not None:
            return slice_x, slice_y
        else:
            return slice_x


class RandomCrop(object):

    def __init__(self, size):
        """
        Randomly crop a torch tensor

        Arguments
        --------
        size : tuple or list
            dimensions of the crop
        """
        self.size = size

    def __call__(self, *inputs):
        h_idx = random.randint(0,inputs[0].size(1)-self.size[0])
        w_idx = random.randint(0,inputs[1].size(2)-self.size[1])
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input[:, h_idx:(h_idx+self.size[0]),w_idx:(w_idx+self.size[1])]
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


class SpecialCrop(object):

    def __init__(self, size, crop_type=0):
        """
        Perform a special crop - one of the four corners or center crop

        Arguments
        ---------
        size : tuple or list
            dimensions of the crop

        crop_type : integer in {0,1,2,3,4}
            0 = center crop
            1 = top left crop
            2 = top right crop
            3 = bottom right crop
            4 = bottom left crop
        """
        if crop_type not in {0, 1, 2, 3, 4}:
            raise ValueError('crop_type must be in {0, 1, 2, 3, 4}')
        self.size = size
        self.crop_type = crop_type
    
    def __call__(self, x, y=None):
        if self.crop_type == 0:
            # center crop
            x_diff  = (x.size(1)-self.size[0])/2.
            y_diff  = (x.size(2)-self.size[1])/2.
            ct_x    = [int(math.ceil(x_diff)),x.size(1)-int(math.floor(x_diff))]
            ct_y    = [int(math.ceil(y_diff)),x.size(2)-int(math.floor(y_diff))]
            indices = [ct_x,ct_y]        
        elif self.crop_type == 1:
            # top left crop
            tl_x = [0, self.size[0]]
            tl_y = [0, self.size[1]]
            indices = [tl_x,tl_y]
        elif self.crop_type == 2:
            # top right crop
            tr_x = [0, self.size[0]]
            tr_y = [x.size(2)-self.size[1], x.size(2)]
            indices = [tr_x,tr_y]
        elif self.crop_type == 3:
            # bottom right crop
            br_x = [x.size(1)-self.size[0],x.size(1)]
            br_y = [x.size(2)-self.size[1],x.size(2)]
            indices = [br_x,br_y]
        elif self.crop_type == 4:
            # bottom left crop
            bl_x = [x.size(1)-self.size[0], x.size(1)]
            bl_y = [0, self.size[1]]
            indices = [bl_x,bl_y]
        
        x = x[:,indices[0][0]:indices[0][1],indices[1][0]:indices[1][1]]

        if y is not None:
            y = y[:,indices[0][0]:indices[0][1],indices[1][0]:indices[1][1]]
            return x, y
        else:
            return x


class Pad(object):

    def __init__(self, size):
        """
        Pads an image to the given size

        Arguments
        ---------
        size : tuple or list
            size of crop
        """
        self.size = size

    def __call__(self, x, y=None):
        x = x.numpy()
        shape_diffs = [int(np.ceil((i_s - d_s))) for d_s,i_s in zip(x.shape,self.size)]
        shape_diffs = np.maximum(shape_diffs,0)
        pad_sizes = [(int(np.ceil(s/2.)),int(np.floor(s/2.))) for s in shape_diffs]
        x = np.pad(x, pad_sizes, mode='constant')
        if y is not None:
            y = y.numpy()
            y = np.pad(y, pad_sizes, mode='constant')
            return tc.from_numpy(x), tc.from_numpy(y)
        else:
            return tc.from_numpy(x)


class RandomFlip(object):

    def __init__(self, h=True, v=False, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.

        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p

        v : boolean
            whether to vertically flip w/ probability p

        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = h
        self.vertical = v
        self.p = p

    def __call__(self, x, y=None):
        x = x.numpy()
        if y is not None:
            y = y.numpy()
        # horizontal flip with p = self.p
        if self.horizontal:
            if random.random() < self.p:
                x = x.swapaxes(2, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 2)
                if y is not None:
                    y = y.swapaxes(2, 0)
                    y = y[::-1, ...]
                    y = y.swapaxes(0, 2)
        # vertical flip with p = self.p
        if self.vertical:
            if random.random() < self.p:
                x = x.swapaxes(1, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 1)
                if y is not None:
                    y = y.swapaxes(1, 0)
                    y = y[::-1, ...]
                    y = y.swapaxes(0, 1)
        if y is None:
            # must copy because torch doesnt current support neg strides
            return tc.from_numpy(x.copy())
        else:
            return tc.from_numpy(x.copy()),tc.from_numpy(y.copy())


class RandomOrder(object):
    """
    Randomly permute the channels of an image
    """
    def __call__(self, *inputs):
        order = tc.randperm(inputs[0].dim())
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.index_select(0, order)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]

