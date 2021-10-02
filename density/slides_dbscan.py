"""

"""
import random


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

        clusters.append(set())
        unclustered.remove(point)

        neighbourhood = find_neighbourhood(
            similarity_matrix, point, radius, unvisited
        )

        # If the density is too low then its just noise
        if len(neighbourhood) < minimum_points:
            noise.append(point)
            unclustered.add(point)
            continue

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
            neighbourhood.add(point)
    return neighbourhood
