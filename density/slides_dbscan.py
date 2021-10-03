"""

"""
import random

import numpy as np
from sklearn.cluster import DBSCAN


def slides_dbscan(similarity_matrix, radius: float, minimum_points: int):
    """
    Does DBSCAN (Density Based Clustering of Applications with Noise)
    Time: O(n^2)
    Space: O(n^2) (pre-existing in similarity matrix)
    :param similarity_matrix: A lookup for distances. Should be able
    to call similarity_matrix[a][b] to be able to find the distance
    between a and b
    :param radius: The radius of a neighbourhood
    :param minimum_points: The minimum number of points for the
    neighbourhood around a given point
    :return: (clusters: list(set), noise: set)
    """
    unvisited = set(range(len(similarity_matrix)))
    unclustered = set(range(len(similarity_matrix)))
    clusters = list()
    noise = list()

    while len(unvisited) > 0:
        point = random.sample(unvisited, 1)[0]
        unvisited.remove(point)

        neighbourhood = find_neighbourhood(
            similarity_matrix, point, radius, unvisited
        )

        # If the density is too low then its just noise
        if len(neighbourhood) < minimum_points:
            noise.append(point)
            unclustered.add(point)
            continue

        clusters.append({point})
        unclustered.remove(point)

        for potential_point in neighbourhood:
            # neighbourhood already only contains unvisited points
            unvisited.remove(potential_point)

            # if epsilon neighbourhood has at least minimum points
            inner_neighbourhood = find_neighbourhood(
                similarity_matrix, point, radius, unvisited
            )

            if len(inner_neighbourhood) > minimum_points:
                neighbourhood.update(inner_neighbourhood)

            if potential_point in unclustered:
                unclustered.remove(potential_point)
                clusters[-1].add(potential_point)

    return clusters, noise


def find_neighbourhood(similarity_matrix, point, radius, unvisited: set):
    neighbourhood = set()
    for u in unvisited:
        if similarity_matrix[point][u] < radius:
            neighbourhood.add(u)
    return neighbourhood


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
