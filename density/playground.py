import numpy as np
from sklearn.cluster import DBSCAN

from density.slides_dbscan import my_DBSCAN

if __name__ == "__main__":
    size = 1000
    np.random.seed(42)
    MOVIES_SIMILARITY_MATRIX = np.random.rand(size, size)
    USERS_SIMILARITY_MATRIX = np.random.rand(size, size)

    MOVIES_RADIUS = 0.001
    MOVIES_MINIMUM_POINTS = 4

    my_dbscan_movies_clustering = my_DBSCAN(
        eps=MOVIES_RADIUS, min_samples=MOVIES_MINIMUM_POINTS, metric='precomputed', n_jobs=-1
    ).fit(MOVIES_SIMILARITY_MATRIX)

    print(set(my_dbscan_movies_clustering.labels_))
