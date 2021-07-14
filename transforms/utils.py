import random
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        if not isinstance(inputs,(list,tuple)):
            inputs = [inputs]
        for transform in self.transforms:
            inputs = transform(inputs)
        return inputs[0] if len(inputs)==1 else inputs


class RandomChoiceCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *inputs):
        tform = random.choice(self.transforms)
        return tform(*inputs)



class OneHot(object):
    def __init__(self, labels):
        self.lb = LabelBinarizer()
        self.lb.fit(labels)
        
    def __call__(self, *inputs):
        return [self.lb.transform(np.array([input]))[0] for input in inputs]



class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, type(lambda: None))
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'