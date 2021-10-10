"""

"""
import math
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from useful_tools import pickle_manager
from numba import njit


def slides_dbscan(feature_vectors, radius: float, minimum_points: int, down_to=0):
    """
    Does DBSCAN (Density Based Clustering of Applications with Noise)
    Time: O(n^2)
    Space: O(n^2) (pre-existing in similarity matrix)
    :param feature_vectors: A lookup for distances. Should be able
    to call similarity_matrix[a][b] to be able to find the distance
    between a and b
    :param radius: The radius of a neighbourhood
    :param minimum_points: The minimum number of points for the
    neighbourhood around a given point
    :return: (clusters: list(set), noise: set)
    """
    # Use unordered sets so we get an O(1) add/remove
    unvisited = set(range(len(feature_vectors)))
    unclustered = set(range(len(feature_vectors)))
    clusters = list()
    noise = list()

    while len(unvisited) > down_to:
        print(len(unvisited))
        point = random.sample(unvisited, 1)[0]
        unvisited.remove(point)

        neighbourhood = find_neighbourhood(
            feature_vectors, point, radius, unvisited
        )

        # If the density is too low then its just noise
        if len(neighbourhood) < minimum_points:
            noise.append(point)
            unclustered.add(point)
            continue

        clusters.append({point})
        unclustered.remove(point)

        for potential_point in neighbourhood:
            if potential_point in unvisited:
                unvisited.remove(potential_point)

            # if epsilon neighbourhood has at least minimum points
            inner_neighbourhood = find_neighbourhood(
                feature_vectors, point, radius, unvisited
            )

            if len(inner_neighbourhood) > minimum_points:
                neighbourhood.update(inner_neighbourhood)

            if potential_point in unclustered:
                unclustered.remove(potential_point)
                clusters[-1].add(potential_point)

    return clusters, noise


def find_neighbourhood(feature_vectors, point, radius, unvisited: set):
    neighbourhood = set()
    for u in unvisited:
        if euclid_distance(feature_vectors[point], feature_vectors[u]) < radius:
            neighbourhood.add(u)
    return neighbourhood


def euclid_distance(v1, v2):
    # If we try to use the non-numpy version, this operation takes about 5
    # seconds per node
    total = 0
    for i in range(len(v1)):
        total += (v1[i] - v2[i]) ** 2
    return total ** (1 / 2)


# This is so that I can just override the fit() function
# for easy evaluation
class my_DBSCAN(DBSCAN):
    """
    Shockingly bad practice to inherit a library class,
    but I'm just going to do it so that the evaluation of my
    algorithm is substantially easier.
    """

    def fit(self, X, y=None, sample_weight=None):
        """
        The dbscan sets the following self variables:
            self.core_sample_indices
            self.labels_
            self.components
        For now I'll just try to set only the labels_ parameter and see what
        happens
        :param X:
        :param y:
        :param sample_weight:
        :return:
        """
        (clusters, noise) = slides_dbscan(
            X, self.eps, self.min_samples
        )

        if not self.eps > 0.0:
            raise ValueError("eps must be positive")
        pass
        self.labels_ = np.zeros(X.shape[0])

        cluster_label = 0
        for cluster in clusters:
            for point in cluster:
                self.labels_[point] = cluster_label
            cluster_label += 1

        for point in noise:
            self.labels_[point] = -1

        return self


if __name__ == '__main__':
    print(os.getcwd())  # This has to be in the root or it'll die

    MOVIES: pd.DataFrame = pickle_manager.load_pickle(
        'pickles/sentences.pickle.lz4'
    )

    MOVIES_FEATURES: pd.DataFrame = pickle_manager.load_pickle(
        'pickles/sentence_features.pickle.lz4'
    )
    MOVIES_FEATURES: np.ndarray = \
        MOVIES_FEATURES.reset_index().to_numpy(dtype=np.float64)

    # Huh apparently we have nan. If we set to 0 then it should be fine
    MOVIES_FEATURES[np.isnan(MOVIES_FEATURES)] = 0
    print("Trying to do 3")
    start = time.perf_counter()
    clusters, noise = slides_dbscan(
        MOVIES_FEATURES, radius=50, minimum_points=100, down_to=len(MOVIES_FEATURES) - 3
    )
    print("That took", time.perf_counter() - start)
