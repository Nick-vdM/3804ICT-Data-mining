import numpy as np
import time
import os
from sklearn.cluster import DBSCAN
from proposal.useful_tools import pickle_manager

from density.slides_dbscan import my_DBSCAN


# %%
def crush(x):
    if x < 0:
        x = 0
    elif x > 1:
        x = 1
    return x


if __name__ == "__main__":
    # Just so that we have something lets just go ahead and do this
    np.random.seed(42)
    os.chdir('..')
    MOVIES_SIMILARITY_MATRIX = pickle_manager.load_pickle('pickles/similarity_matrix.pickle.lz4')
    MOVIES_RADIUS = 0.8
    MOVIES_MINIMUM_POINTS = 5

    crush_v = np.vectorize(crush)
    MOVIES_SIMILARITY_MATRIX = crush_v(np.subtract(1.0, MOVIES_SIMILARITY_MATRIX))
    print(MOVIES_SIMILARITY_MATRIX)
    start = time.perf_counter()
    my_dbscan_movies_clustering = my_DBSCAN(
        eps=MOVIES_RADIUS, min_samples=MOVIES_MINIMUM_POINTS, metric='precomputed', n_jobs=1
    ).fit(MOVIES_SIMILARITY_MATRIX)
    print("Took", time.perf_counter() - start, "seconds")

    print(set(my_dbscan_movies_clustering.labels_))
    ## Evaluation
