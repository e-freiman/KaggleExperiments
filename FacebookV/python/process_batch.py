import numpy as np
import pandas as pd
from os.path import join
import time
# NumPy and C++ implementation
import process_event

# For measuring elapsing time
GAP = 0
counter = 0
start = 0
finish = 0
totally = 0


def compute_ids(row, model):

    global counter
    global GAP
    global start
    global finish
    global totally

    if counter % GAP == 0:
        finish = time.time()
        if counter > 0:
            velocity = GAP / (finish - start)
            print(round(velocity, 1), 'id/sec', round((totally - counter) / velocity / 60 / 60, 2), 'hours left')
        start = time.time()

    counter += 1

    # id = row[0]
    x = row[1]
    y = row[2]
    accuracy = row[3]
    # raw time = row[4]
    day_of_week = row[5]
    time_of_day = row[6]

    # NumPy and C++ implementation
    return process_event.compute(x, y, accuracy, day_of_week, time_of_day, model)


def read_meta(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return [float(lines[0]), float(lines[1]), int(lines[2])]


def compute(args):

    # This procedure works once for an every computational core

    data = args[0]
    model_dir = args[1]
    global GAP
    GAP = args[2]

    print('--- compute, data.shape =', data.shape, '---')

    global totally
    totally = data.shape[0]

    # Loading model
    class Model:
        pass

    model = Model()
    prior_probabilities = pd.read_csv(join(model_dir, 'prior_probabilities.csv'), dtype={'id': str})
    model.prior = np.array(prior_probabilities['prob'], np.float32)
    model.labels = prior_probabilities['id']

    gauss_location = pd.read_csv(join(model_dir, 'gauss_location.csv'),
                                 dtype={'id': np.int64, 'mu1': np.float32, 'mu2': np.float32,
                                        's1': np.float32, 's12': np.float32, 's2': np.float32})

    model.hist_days = pd.read_csv(join(model_dir, 'hist_day_of_week.csv'), dtype=np.float32).as_matrix()[:, 1:]
    model.hist_time = pd.read_csv(join(model_dir, 'hist_time_of_day.csv'), dtype=np.float32).as_matrix()[:, 1:]
    model.hist_accuracy = pd.read_csv(join(model_dir, 'hist_accuracy.csv'), dtype=np.float32).as_matrix()[:, 1:]

    model.bins_days = read_meta(join(model_dir, 'hist_day_of_week.meta'))
    model.bins_time = read_meta(join(model_dir, 'hist_time_of_day.meta'))
    model.bins_accuracy = read_meta(join(model_dir, 'hist_accuracy.meta'))

    model.gauss_location = gauss_location.as_matrix()

    return list(np.apply_along_axis(compute_ids, axis=1,  arr=data, model=model))
