import numpy as np


def weighted_kappa(label, pred, min_rating=None, max_rating=None):
    label = np.array(label, dtype=int)
    pred = np.array(pred, dtype=int)
    assert(len(label) == len(pred))
    if min_rating is None:
        min_rating = min(min(label), min(pred))
    if max_rating is None:
        max_rating = max(max(label), max(pred))
    conf_mat = confusion_matrix_kappa(label, pred, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(label))

    hist_label = histogram(label, min_rating, max_rating)
    hist_pred = histogram(pred, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_label[i] * hist_pred[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / (pow(num_ratings - 1, 2.0)+1E-10)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / (denominator+1E-10)

def confusion_matrix_kappa(label, pred, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """

    assert(len(label) == len(pred))
    if min_rating is None:
        min_rating = min(label + pred)
    if max_rating is None:
        max_rating = max(label + pred)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(label, pred):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings
