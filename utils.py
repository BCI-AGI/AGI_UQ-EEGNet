import pickle
import random
import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt


def data_load(subjs):
    X_list, y_list = [], []
    for s in subjs:
        with open('./data/subj_{}.pkl'.format(s), 'rb') as f:
            data = pickle.load(f)
        epochs, labels = data['epochs'], data['labels']
        X_list.append(epochs)
        y_list.append(labels)
    return X_list, y_list

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def make_Tensor(array):
    return torch.from_numpy(array).float()

def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    layer.reset_parameters()

def train_valid_split(X, y, test_size):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    sss.get_n_splits(X, y)
    for train_idx, test_idx in sss.split(X, y):
        X_train, y_train = np.array(X)[train_idx], np.array(y)[train_idx]
        X_val, y_val = np.array(X)[test_idx], np.array(y)[test_idx]
    return X_train, y_train, X_val, y_val


def expected_calibration_error(samples, true_labels, M=10):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece
