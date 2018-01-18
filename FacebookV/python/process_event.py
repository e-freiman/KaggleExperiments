import numpy as np
import platform
from ctypes import CDLL, c_double

if any(platform.win32_ver()):
    gaussDll = CDLL("cpp/gauss.dll")
else:
    gaussDll = CDLL("cpp/gauss.o")

gaussDll.gauss.restype = c_double
gaussDll.gauss.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_double]


def compute_hist(prob, bins, x):
    lo = bins[0]
    hi = bins[1]
    n = bins[2]
    pos = int((x - lo) / (hi - lo) * (n - 1))
    return prob[pos]


def compute_gauss(row, x1, x2):
    mu1 = row[2]
    mu2 = row[3]
    s1 = row[4]
    s12 = row[5]
    s2 = row[6]
    return gaussDll.gauss(mu1, mu2, s1, s12, s2, x1, x2)


# Numpy and C++ implementation
def compute(x, y, accuracy, day_of_week, time_of_day, model):
    prob_location = np.apply_along_axis(compute_gauss, axis=1,  arr=model.gauss_location, x1=x, x2=y)
    prob_day = np.apply_along_axis(compute_hist, axis=1, arr=model.hist_days, bins=model.bins_days, x=day_of_week)
    prob_time = np.apply_along_axis(compute_hist, axis=1, arr=model.hist_time, bins=model.bins_time, x=time_of_day)
    prob_accuracy = np.apply_along_axis(compute_hist, axis=1, arr=model.hist_accuracy, bins=model.bins_accuracy,
                                        x=accuracy)

    prob = model.prior * prob_location * prob_day * prob_time * prob_accuracy
    sorted_prob_indexes = np.argsort(prob)

    id1 = model.labels[sorted_prob_indexes[len(sorted_prob_indexes)-1]]
    id2 = model.labels[sorted_prob_indexes[len(sorted_prob_indexes)-2]]
    id3 = model.labels[sorted_prob_indexes[len(sorted_prob_indexes)-3]]

    return id1+' '+id2+' '+id3
