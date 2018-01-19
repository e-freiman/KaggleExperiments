import pandas as pd
import sys
import process_batch
from multiprocessing import Pool, cpu_count
from itertools import chain


# Features engineering (decoding time)
def decode_time(data):
    minutes_in_day = 60 * 24
    day_of_week = (data['time'] // minutes_in_day) % 7
    time_of_day = data['time'] % minutes_in_day

    data['day_of_week'] = day_of_week
    data['time_of_day'] = time_of_day

    return data

if __name__ == '__main__':

    # Loading data
    if len(sys.argv) != 4:
        sys.exit('Wrong number of parameters provided.'
                 ' It should be "facebook_v.py <path to the model directory> <path to the data> <visualization gap>"')

    model_dir = sys.argv[1]
    data_path = sys.argv[2]
    visualization_gap = int(sys.argv[3])

    data = decode_time(pd.read_csv(data_path))

    if 'place_id' in data.columns:
        place_id_column = data['place_id']
        del data['place_id']
    else:
        place_id_column = None

    # Calculating data subsets
    N = cpu_count()
    size = int(data.shape[0] / N)
    subsets_indexes = [(i * size, (i + 1) * size - 1) for i in range(N-1)] + [((N - 1) * size, data.shape[0] - 1)]

    print(subsets_indexes)

    # Parallel predictions computation
    results = Pool().map_async(process_batch.compute, ((data.loc[idx[0]:idx[1]], model_dir, visualization_gap) for idx in subsets_indexes)).get()
    merged_results = list(chain.from_iterable(results))

    # Serialized version of computations for profiling
    # merged_results = process_batch.compute((data, model_dir, visualization_gap))

    # save results to csv
    df_result = pd.DataFrame(list(merged_results), columns=['place_id'])
    df_result.index.name = 'row_id'
    df_result.to_csv('result.csv')

    print('!!! Results saved to "result.csv" !!!')

    # Test on provided labels (uses test or train data depending on how we get our model)
    if place_id_column is not None:
        data['computed_result'] = merged_results
        data['place_id'] = place_id_column

        def compute_score(row):
            computed_ids = row['computed_result'].split(' ')
            place_id = str(row['place_id'])
            return place_id in computed_ids

        score = sum(data.apply(compute_score, axis=1))/data.shape[0]
        print('Success guesses', round(score * 100, 3), '%')