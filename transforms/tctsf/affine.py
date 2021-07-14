"""
Affine transforms implemented on torch tensors, and
requiring only one interpolation
"""
import random
import torch as tc
from .util1 import get_zoom_matrix, get_shear_matrix, get_translate_matrix, apply_affine, get_rotate_matrix
from .util2 import tc_random_choice

class RandomAffine(object):

    def __init__(self, 
                 rotation_range=None, 
                 translation_range=None,
                 shear_range=None, 
                 zoom_range=None,
                 interp='bilinear'):
        """
        Perform an affine transforms with various sub-transforms, using
        only one interpolation and without having to instantiate each
        sub-transform individually.

        Arguments
        ---------
        rotation_range : one integer or float
            image will be rotated randomly between (-degrees, degrees) 

        translation_range : a float or a tuple/list with 2 floats between [0, 1)
            first value:
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)

        shear_range : float
            image will be sheared randomly between (-degrees, degrees)

        zoom_range : list/tuple with two floats between [0, infinity).
            first float should be less than the second
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out

        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']

        """
        self.transforms = []
        if rotation_range is not None:
            rotation_tform = RandomRotate(rotation_range, lazy=True)
            self.transforms.append(rotation_tform)

        if translation_range is not None:
            translation_tform = RandomTranslate(translation_range, lazy=True)
            self.transforms.append(translation_tform)

        if shear_range is not None:
            shear_tform = RandomShear(shear_range, lazy=True)
            self.transforms.append(shear_tform)

        if zoom_range is not None:
            zoom_tform = RandomZoom(zoom_range, lazy=True)
            self.transforms.append(zoom_tform)

        self.interp = interp

        if len(self.transforms) == 0:
            raise Exception('Must give at least one transform parameter')

    def __call__(self, *inputs):
        matrix = self.transforms[0](inputs[0])
        for tform in self.transforms[1:]:
            matrix = matrix.mm(tform(inputs[0]))

        outputs = apply_affine(*inputs, matrix, interp=self.interp)
        return outputs


class RandomRotate(object):

    def __init__(self, 
                 rotation_range,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees

        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']

        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        self.rotation_range = rotation_range
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        matrix = get_rotate_matrix(angle)
        if self.lazy:
            return matrix
        else:
            outputs = apply_affine(*inputs, matrix, interp=self.interp)
            return outputs


class RandomChoiceRotate(object):

    def __init__(self,  angles,
                        p=None,
                        interp='bilinear',
                        lazy=False):
        """
        Randomly rotate an image from a list of values. If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        values : a list or tuple
            the values from which the rotation value will be sampled

        p : a list or tuple the same length as `values`
            the probabilities of sampling any given value. Must sum to 1.
        """
        if isinstance(angles, (list, tuple)):
            angles = tc.FloatTensor(angles)
        self.angles = angles
        if p is None:
            p = tc.ones(len(angles)) / len(angles)
        else:
            if abs(1.0-sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        angle = tc_random_choice(self.angles, p=self.p)
        matrix = get_rotate_matrix(angle)
        if self.lazy:
            return matrix
        else:
            outputs = apply_affine(*inputs, matrix, interp=self.interp)
            return outputs





class RandomTranslate(object):

    def __init__(self, ratio_hw, interp='bilinear', lazy=False):
        """
        Randomly translate an image some fraction of total height and/or
        some fraction of total widtc. If the image has multiple channels,
        the same translation will be applied to each channel.

        Arguments
        ---------
        translation_range : two floats between [0, 1) 
            first value:
                fractional bounds of total height to shift image
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                fractional bounds of total width to shift image 
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)
        """
        if isinstance(ratio_hw, float):
            ratio_hw = (ratio_hw, ratio_hw)
        assert(-1<ratio_hw[0] and ratio_hw[0]<1)
        assert(-1<ratio_hw[1] and ratio_hw[1]<1)
        self.ratio_hw = ratio_hw
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        hr = random.uniform(-self.ratio_hw[0], self.ratio_hw[0])
        wr = random.uniform(-self.ratio_hw[1], self.ratio_hw[1])
        tsl_h = hr*inputs[0].size(1)
        tsl_w = wr*inputs[0].size(2)
        matrix = get_translate_matrix(tsl_h, tsl_w)
        if self.lazy:
            return matrix
        else:
            outputs = apply_affine(*inputs, matrix, interp=self.interp)
            return outputs


class RandomChoiceTranslate(object):

    def __init__(self, ratios, p=None, interp='bilinear', lazy=False):
        """
        Randomly translate an image some fraction of total height and/or
        some fraction of total width from a list of potential values. 
        If the image has multiple channels,
        the same translation will be applied to each channel.

        Arguments
        ---------
        values : a list or tuple
            the values from which the translation value will be sampled

        p : a list or tuple the same length as `values`
            the probabilities of sampling any given value. Must sum to 1.

        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']

        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        if isinstance(ratios, (list, tuple)):
            ratios = tc.FloatTensor(ratios)
        self.ratios = ratios
        if p is None:
            p = tc.ones(len(ratios)) / len(ratios)
        else:
            if abs(1.0-sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        hr = tc_random_choice(self.ratios, p=self.p)
        wr = tc_random_choice(self.ratios, p=self.p)
        tsl_h = hr*inputs[0].size(1)
        tsl_w = wr*inputs[0].size(2)
        matrix = get_translate_matrix(tsl_h, tsl_w)
        if self.lazy:
            return matrix
        else:
            outputs = apply_affine(*inputs, matrix, interp=self.interp)
            return outputs


class RandomShear(object):

    def __init__(self, angle,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly shear an image with radians (-shear_range, shear_range)
        """
        self.angle = angle
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        angle = random.uniform(-self.angle, self.angle)
        matrix = get_shear_matrix(angle)
        if self.lazy:
            return matrix
        else:
            outputs = apply_affine(*inputs, matrix, interp=self.interp)
            return outputs
        

class RandomChoiceShear(object):

    def __init__(self, angles, p=None, interp='bilinear', lazy=False):
        
        if isinstance(angles, (list, tuple)):
            angles = tc.FloatTensor(angles)
        self.angles = angles
        if p is None:
            p = tc.ones(len(angles)) / len(angles)
        else:
            if abs(1.0-sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        angle = tc_random_choice(self.angles, p=self.p)
        matrix = get_shear_matrix(angle)
        if self.lazy:
            return matrix
        else:
            outputs = apply_affine(*inputs, matrix, interp=self.interp)
            return outputs


class RandomZoom(object):

    def __init__(self, zoom_range, interp='bilinear', lazy=False):
        """
        Randomly zoom in and/or out on an image 

        Arguments
        ---------
        zoom_range : tuple or list with 2 values, both between (0, infinity)
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out
        """
        if not isinstance(zoom_range, list) and not isinstance(zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        self.zoom_range = zoom_range
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zy = random.uniform(self.zoom_range[0], self.zoom_range[1])
        matrix = get_zoom_matrix(zx, zy)
        if self.lazy:
            return matrix
        else:
            outputs = apply_affine(*inputs, matrix, interp=self.interp)
            return outputs


class RandomChoiceZoom(object):

    def __init__(self, 
                 values,
                 p=None,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly zoom in and/or out on an image with a value sampled from
        a list of values

        Arguments
        ---------
        values : a list or tuple
            the values from which the applied zoom value will be sampled

        p : a list or tuple the same length as `values`
            the probabilities of sampling any given value. Must sum to 1. 
        """
        if isinstance(values, (list, tuple)):
            values = tc.FloatTensor(values)
        self.values = values
        if p is None:
            p = tc.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        zx = tc_random_choice(self.values, p=self.p)
        zy = tc_random_choice(self.values, p=self.p)
        matrix = get_zoom_matrix(zx, zy)
        if self.lazy:
            return matrix
        else:
            outputs = apply_affine(*inputs, matrix, interp=self.interp)
            return outputs





