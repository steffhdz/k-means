import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from Point import Point
from Cluster import Cluster

DATASET1 = "./dataSet/DS_3Clusters_999Points.txt"
DATASET2 = "./dataSet/DS2_3Clusters_999Points.txt"
DATASET3 = "./dataSet/DS_5Clusters_10000Points.txt"
DATASET4 = "./dataSet/DS_7Clusters_100000Points.txt"
NUM_CLUSTERS = 3
ITERATIONS = 1000
COLORS = ['red', 'blue', 'green', 'yellow', 'gray', 'pink', 'violet', 'brown',
        'cyan', 'magenta']


def dataset_to_list_points(dir_dataset):

    points = list()
    with open(dir_dataset, 'rt') as reader:
        for point in reader:
            points.append(Point(np.asarray(map(float, point.split("::")))))
    return points


def get_nearest_cluster(clusters, point):

    dist = np.zeros(len(clusters))
    for i, c in enumerate(clusters):
        dist[i] = distance.euclidean(point.coordinates, c.centroid)
    return np.argmin(dist)


def print_clusters_status(it_counter, clusters):
    print ('\nITERATION %d') % it_counter
    for i, c in enumerate(clusters):
        print ('\tCentroid Cluster %d: %s') % (i + 1, str(c.centroid))


def print_results(clusters):
    print ('\n\nFINAL RESULT:')
    for i, c in enumerate(clusters):
        print ('\tCluster %d') % (i + 1)
        print ('\t\tNumber Points in Cluster %d') % len(c.points)
        print ('\t\tCentroid: %s') % str(c.centroid)


def plot_results(clusters):
    plt.plot()
    for i, c in enumerate(clusters):
        # plot points
        x, y = zip(*[p.coordinates for p in c.points])
        plt.plot(x, y, linestyle='None', color=COLORS[i], marker='.')
        # plot centroids
        plt.plot(c.centroid[0], c.centroid[1], 'o', color=COLORS[i],
                markeredgecolor='k', markersize=10)
    plt.show()


def k_means(dataset, num_clusters, iterations):
    # Read data set
    points = dataset_to_list_points(dataset)

    # Select N points random to initiacize the N Clusters
    initial = random.sample(points, num_clusters)

    # Create N initial Clusters
    clusters = [Cluster([p]) for p in initial]

    # Inicialize list of lists to save the new points of cluster
    new_points_cluster = [[] for i in range(num_clusters)]

    converge = False
    it_counter = 0
    while (not converge) and (it_counter < iterations):
        # Assign points in nearest centroid
        for p in points:
            i_cluster = get_nearest_cluster(clusters, p)
            new_points_cluster[i_cluster].append(p)

        # Set new points in clusters and calculate de new centroids
        for i, c in enumerate(clusters):
            c.update_cluster(new_points_cluster[i])

        # Check that converge all Clusters
        converge = [c.converge for c in clusters].count(False) == 0

        # Increment counter and delete lists of clusters points
        it_counter += 1
        new_points_cluster = [[] for i in range(num_clusters)]

        # Print clusters status
        print_clusters_status(it_counter, clusters)

    # Print final result
    print_results(clusters)

    # Plot Final results
    plot_results(clusters)


if __name__ == '__main__':
    k_means(DATASET1, NUM_CLUSTERS, ITERATIONS)