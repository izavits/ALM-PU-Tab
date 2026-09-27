"""
PU data preparation and evaluation.
"""
import math

import numpy as np
from pandas import read_csv
from sklearn.metrics import (accuracy_score, average_precision_score, pairwise_distances, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_consistent_length


def load_csv(path):
    df = read_csv(path, header=None)
    X = df.values[:, :-1].astype('float32')
    y = df.values[:, -1].astype('float32')
    return X, y.reshape((len(y), 1))


def get_splits(X, y, n_test=0.30, random_state=0):
    return train_test_split(np.arange(len(X)), test_size=n_test, random_state=random_state,
                            shuffle=True, stratify=y)


def create_pu_data(X, y, ratio=None, label_assumption='SCAR', n_neighbors=None, random_state=123):
    """Hide `ratio` of the positives (set them to 0); SAR picks them with probability proportional
    to their mean distance to the first `n_neighbors` negatives (in row order, as in pu_data.py)."""
    y_ = y.copy()
    check_consistent_length(X, y_)
    n_pos = np.sum(y_ == 1)
    random_state = check_random_state(random_state)

    if ratio is None:
        return y_
    if n_neighbors is None:
        n_neighbors = math.floor(math.sqrt(len(y_)))
    if label_assumption == 'SCAR':
        p_ = None
    elif label_assumption == 'SAR':
        dist_matrix = pairwise_distances(X, n_jobs=-1)
        ix_pos = np.where(y == 1)[0]
        ix_neg = np.where(y == 0)[0]
        pos_dist_mt = dist_matrix[ix_pos][:, ix_neg]
        k_mean_dist = np.mean(pos_dist_mt[:, :n_neighbors], axis=1)
        sample_weight = np.array([k_mean_dist[i] / np.sum(k_mean_dist) for i, _ in enumerate(k_mean_dist)])
        p_ = sample_weight.reshape(-1, )
    else:
        raise ValueError('Labeling mechanism not valid; Use SCAR or SAR.')

    if label_assumption == 'SAR':
        size_ = min(int(np.ceil(ratio * n_pos)), len(np.nonzero(p_)[0]))
    else:
        size_ = int(np.ceil(ratio * n_pos))
    indices = random_state.choice(range(n_pos), size_, replace=False, p=p_)
    pos_y = y_[y_ == 1].copy()
    pos_y[indices] = 0
    y_[y_ == 1] = pos_y
    return y_


def prepare_pu_data_plain(path, pu_ratio, labeling_mechanism, random_state):
    X, y = load_csv(path)
    train_idx, test_idx = get_splits(X, y, random_state=random_state)
    x_train, y_train = X[train_idx], y[train_idx]
    loss_prior = np.mean(y_train)
    x_test, y_test = X[test_idx], y[test_idx]
    y_ = create_pu_data(x_train, y_train, pu_ratio, labeling_mechanism, None)
    return x_train, y_, x_test, y_test, loss_prior, y_train


def rank_at_n(n, y_pred_proba, y_test):
    top_n = np.argsort(y_pred_proba)[::-1][:n]
    correct = sum(1 for index in top_n if y_test[index] == 1)
    all_in_test_pos = sum(1 for i in y_test if i == 1)
    return float(correct / n), float(correct / all_in_test_pos)


def r_precision_at_n(y_pred_proba, y_test):
    num_positives_in_test = sum(1 for y in y_test if y == 1)
    precision, _ = rank_at_n(num_positives_in_test, y_pred_proba, y_test)
    return precision


def validate_scores(y_test, pred_proba):
    predictions = [y_hat.round() for y_hat in pred_proba]
    acc = accuracy_score(y_test, predictions)
    area = roc_auc_score(y_test, pred_proba)
    rprecision = r_precision_at_n(pred_proba, y_test)
    precision = precision_score(y_test, predictions, average='binary', zero_division=0)
    recall = recall_score(y_test, predictions, average='binary', zero_division=0)
    pr_area = average_precision_score(y_test, pred_proba)
    p, r, _ = precision_recall_curve(y_test, pred_proba)
    numerator = 2 * r * p
    denom = r + p
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    best_f1 = np.max(f1_scores)
    return {'acc': acc, 'roc_auc': area, 'rprecision': rprecision, 'precision': precision,
            'recall': recall, 'ap': pr_area, 'f1': best_f1}
