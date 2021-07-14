"""
Transforms very specific to images such as 
color, lighting, contrast, brightness, etc transforms

NOTE: Most of these transforms:
1. assume your image intensity is between 0 and 1,
2. are torch tensors (NOT numpy or PIL)
3. channel first (img_ch, img_h, img_w)
"""

import random
import torch as tc
from Ailib.transforms.tctsf import tc_random_choice


def rgb2gray(*inputs, weights=[0.333,0.333,0.333], img_ch=1):
    outputs = []
    for idx, input in enumerate(inputs):
        output = input[0]*weights[0] + input[1]*weights[1] + input[2]*weights[2]
        outputs.append(output.repeat(img_ch,1,1))
    return outputs

def adjustgamma(*inputs, ratio=1):
    return [tc.pow(input, ratio) for input in inputs]

def brightness(*inputs, value):
    """
    Alter the Brightness of an image
    Arguments
    ---------
    value : brightness factor
        =-1 = completely black
        <0 = darker
        0 = no change
        >0 = brighter
        =1 = completely white
    """
    value = max(min(value,1.0),-1.0)
    return [tc.clamp(input.float().add(value).type(input.type()), 0, 1) for input in inputs]

def saturation(*inputs, value):
    """
    Alter the Saturation of image

    Arguments
    ---------
    value : float
        =-1 : gray
        <0 : colors are more muted
        =0 : image stays the same
        >0 : colors are more pure
        =1 : most saturated
    """
    value = max(min(value,1.0),-1.0)

    outputs = []
    for idx, input in enumerate(inputs):
        gray = rgb2gray(input, img_ch=3)
        alpha = 1.0 + value
        output = input.mul(alpha).add(1 - alpha, gray)
        output = tc.clamp(output, 0, 1)
        outputs.append(output)
    return outputs


def contrast(*inputs, value):
    """
    Adjust Contrast of image.

    Contrast is adjusted independently for each channel of each image.

    For each channel, this Op computes the mean of the image pixels 
    in the channel and then adjusts each component x of each pixel to 
    (x - mean) * contrast_factor + mean.

    Arguments
    ---------
    value : float
        smaller value: less contrast
        ZERO: channel means
        larger positive value: greater contrast
        larger negative value: greater inverse contrast
    """
    outputs = []
    for idx, input in enumerate(inputs):
        channel_means = input.mean(1).mean(2)
        channel_means = channel_means.expand_as(input)
        output = tc.clamp((input - channel_means) * value + channel_means,0,1)
        outputs.append(output)
    return outputs

# ----------------------------------------------------
# ----------------------------------------------------
class RandomGrayscale(object):

    def __init__(self, weigths, p=0.5):
        """
        Randomly convert RGB image(s) to Grayscale w/ some probability
        p : a float
            probability that image will be grayscaled
        """
        self.weights=weigths
        self.p = p

    def __call__(self, *inputs):
        pval = random.random()
        if pval < self.p:
            outputs = rgb2gray(*inputs, weights=self.weights, img_ch=3)
        else:
            outputs = inputs
        return outputs


# ----------------------------------------------------
# ----------------------------------------------------

class RandomGamma(object):

    def __init__(self, min_val, max_val):
        """
        Performs Gamma Correction on the input image with some
        randomly selected gamma value between min_val and max_val. 
        Also known as Power Law Transform. This function transforms 
        the input image pixelwise according to the equation 
        Out = In**gamma after scaling each pixel to the range 0 to 1.

        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range

        NOTE:
        for values:
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = adjustgamma(*inputs, value=value)
        return outputs


class RandomChoiceGamma(object):
    def __init__(self, values, p=None):
        """
        Performs Gamma Correction on the input image with some
        gamma value selected in the list of given values.
        Also known as Power Law Transform. This function transforms 
        the input image pixelwise according to the equation 
        Out = In**gamma after scaling each pixel to the range 0 to 1.

        Arguments
        ---------
        values : list of floats
            gamma values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.

        NOTE:
        for values:
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = tc_random_choice(self.values, p=self.p)
        outputs = adjustgamma(*inputs, value=value)
        return outputs

# ----------------------------------------------------
# ----------------------------------------------------



class RandomBrightness(object):

    def __init__(self, min_val, max_val):
        """
        Alter the Brightness of an image with a value randomly selected
        between `min_val` and `max_val`

        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = brightness(*inputs, value=value)
        return outputs

class RandomChoiceBrightness(object):

    def __init__(self, values, p=None):
        """
        Alter the Brightness of an image with a value randomly selected
        from the list of given values with given probabilities

        Arguments
        ---------
        values : list of floats
            brightness values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.
        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = tc_random_choice(self.values, p=self.p)
        outputs = brightness(*inputs, value=value)
        return outputs

# ----------------------------------------------------
# ----------------------------------------------------

class RandomSaturation(object):

    def __init__(self, min_val, max_val):
        """
        Alter the Saturation of an image with a value randomly selected
        between `min_val` and `max_val`

        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = saturation(*inputs, value=value)
        return outputs

class RandomChoiceSaturation(object):

    def __init__(self, values, p=None):
        """
        Alter the Saturation of an image with a value randomly selected
        from the list of given values with given probabilities

        Arguments
        ---------
        values : list of floats
            saturation values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.

        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = tc_random_choice(self.values, p=self.p)
        outputs = saturation(*inputs, value=value)
        return outputs

# ----------------------------------------------------
# ----------------------------------------------------

class RandomContrast(object):

    def __init__(self, min_val, max_val):
        """
        Alter the Contrast of an image with a value randomly selected
        between `min_val` and `max_val`

        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = contrast(*inputs, value=value)
        return outputs

class RandomChoiceContrast(object):

    def __init__(self, values, p=None):
        """
        Alter the Contrast of an image with a value randomly selected
        from the list of given values with given probabilities

        Arguments
        ---------
        values : list of floats
            contrast values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.

        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = tc_random_choice(self.values, p=None)
        outputs = contrast(*inputs, value=value)
        return outputs

# ----------------------------------------------------
# ----------------------------------------------------

def rgb_to_hsv(x):
    """
    Convert from RGB to HSV
    """
    hsv = tc.zeros(*x.size())
    c_min = x.min(0)
    c_max = x.max(0)

    delta = c_max[0] - c_min[0]

    # set H
    r_idx = c_max[1].eq(0)
    hsv[0][r_idx] = ((x[1][r_idx] - x[2][r_idx]) / delta[r_idx]) % 6
    g_idx = c_max[1].eq(1)
    hsv[0][g_idx] = 2 + ((x[2][g_idx] - x[0][g_idx]) / delta[g_idx])
    b_idx = c_max[1].eq(2)
    hsv[0][b_idx] = 4 + ((x[0][b_idx] - x[1][b_idx]) / delta[b_idx])
    hsv[0] = hsv[0].mul(60)

    # set S
    hsv[1] = delta / c_max[0]

    # set V - good
    hsv[2] = c_max[0]

    return hsv
