
from __future__ import absolute_import
from __future__ import print_function

import torch as tc
from ..tools import tensorlib as tl
from ...eval import plot_confusion_matrix, weighted_kappa


class Metric(object):

    def __call__(self, y_pred, y_true):
        raise NotImplementedError('Custom Metrics must implement this function')

    def reset(self):
        raise NotImplementedError('Custom Metrics must implement this function')

from sklearn.metrics import confusion_matrix

class Kappa(Metric):

    def __init__(self, fig_path, classes, is_normalize=True, name='Kappa'):
        self.y_true = []
        self.y_pred = []
        self.name = name
        self.fig_path = fig_path
        self.is_normalize = is_normalize
        self.classes = classes

    def reset(self):
        if len(self.y_true)>0:
            confusion_mat = confusion_matrix(self.y_true, self.y_pred)
            plot_confusion_matrix(confusion_mat, self.fig_path, self.is_normalize, self.classes)
        self.y_true = []
        self.y_pred = []

    def __call__(self, y_pred, y_true, min_rating=None, max_rating=None):
        self.y_true.extend(y_true.cpu().tolist())
        self.y_pred.extend(y_pred.cpu().max(1)[1].tolist())
        kappa = weighted_kappa(self.y_true, self.y_pred)
        return kappa



class CategoricalAccuracy(Metric):

    def __init__(self, top_k=1):
        self.top_k = top_k
        self.correct_count = 0
        self.total_count = 0

        self.name = 'acc'

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, y_pred, y_true):
        top_k = y_pred.topk(self.top_k,1)[1]
        true_k = y_true.view(len(y_true),1).expand_as(top_k)
        self.correct_count += top_k.eq(true_k).float().sum().data[0]
        self.total_count += len(y_pred)
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        return accuracy


class BinaryAccuracy(Metric):

    def __init__(self):
        self.correct_count = 0
        self.total_count = 0

        self.name = 'acc_metric'

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, y_pred, y_true):
        y_pred_round = y_pred.round().long()
        self.correct_count += y_pred_round.eq(y_true).float().sum().data[0]
        self.total_count += len(y_pred)
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        return accuracy


class ProjectionCorrelation(Metric):

    def __init__(self):
        self.corr_sum = 0.
        self.total_count = 0.

        self.name = 'corr_metric'

    def reset(self):
        self.corr_sum = 0.
        self.total_count = 0.

    def __call__(self, y_pred, y_true=None):
        """
        y_pred should be two projections
        """
        covar_mat = tc.abs(tl.tc_matrixcorr(y_pred[0].data, y_pred[1].data))
        self.corr_sum += tc.trace(covar_mat)
        self.total_count += covar_mat.size(0)
        return self.corr_sum / self.total_count


class ProjectionAntiCorrelation(Metric):

    def __init__(self):
        self.anticorr_sum = 0.
        self.total_count = 0.

        self.name = 'anticorr_metric'

    def reset(self):
        self.anticorr_sum = 0.
        self.total_count = 0.

    def __call__(self, y_pred, y_true=None):
        """
        y_pred should be two projections
        """
        covar_mat = tc.abs(tl.tc_matrixcorr(y_pred[0].data, y_pred[1].data))
        upper_sum = tc.sum(tc.triu(covar_mat,1))
        lower_sum = tc.sum(tc.tril(covar_mat,-1))
        self.anticorr_sum += upper_sum
        self.anticorr_sum += lower_sum
        self.total_count += covar_mat.size(0)*(covar_mat.size(1) - 1)
        return self.anticorr_sum / self.total_count



