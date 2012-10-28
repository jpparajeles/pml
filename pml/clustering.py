# Copyright (C) 2012 David Rusk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to 
# deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
# IN THE SOFTWARE.
"""
Clustering algorithms for unsupervised learning tasks.

@author: drusk
"""

import random

import pandas as pd

import model
from distance_utils import euclidean
from pandas_util import are_dataframes_equal

def create_random_centroids(dataset, k):
    """
    Initializes centroids at random positions.
    
    The random value chosen for each feature will always be limited to the 
    range of values found in the dataset.  For example, if a certain feature 
    has a minimum value of 0 in the dataset, and maximum value of 9, the
    
    Args:
      dataset: DataSet
        The DataSet to create the random centroids for.
      k: int
        The number of centroids to create.
        
    Returns:
      A list of centroids.  Each centroid is a pandas Series with the same 
      labels as the dataset's headers.
    """
    min_maxs = zip(dataset.reduce_features(min).values, 
                   dataset.reduce_features(max).values)

    def rand_range(range_tuple):
        """
        Generates a random floating point number in the range specified by 
        the tuple.
        """
        return random.uniform(range_tuple[0], range_tuple[1])

    return [pd.Series(map(rand_range, min_maxs), index=dataset.feature_list(), 
                      name = i) for i in range(k)]

def kmeans(dataset, k=2, create_centroids=create_random_centroids):
    """
    K-means clustering algorithm.
    
    This algorithm partitions a dataset into k clusters in which each 
    observation (sample) belongs to the cluster with the nearest mean.
    
    Args:
      dataset: model.DataSet
        The DataSet to perform the clustering on.
      k: int
        The number of clusters to partition the dataset into.
      create_centroids: function
        The function specifying how to create the initial centroids for the 
        clusters.  Defaults to creating them randomly.
        
    Returns:
      A pandas Series.  Each sample index is assigned a numerical value 
      representing the cluster it is part of.
    """
    # If dataset is not already a model.DataSet object, make it one.
    dataset = model.as_dataset(dataset)
    
    # Initialize k centroids
    centroids = create_centroids(dataset, k)
    
    # Iteratively compute best clusters until they stabilize
    assignments = None
    clusters_changed = True
    while clusters_changed:
        centroids, new_assignments = _compute_iteration(dataset, centroids)
        if are_dataframes_equal(new_assignments, assignments):
            clusters_changed = False
        assignments = new_assignments
    
    return assignments

def _get_distances_to_centroids(dataset, centroids):
    """
    Calculates the calc_distance from each data point to each centroid.
    
    Args:
      dataset: model.DataSet
        The DataSet whose samples are being 
      centroids: list of pandas Series
        The centroids to compare each data point with.
        
    Returns:
      A pandas DataFrame with a row for each sample in dataset and a column 
      for the distance to each centroid.
    """
    distances = {}
    for i, centroid in enumerate(centroids):
        def calc_distance(sample):
            # TODO: parameter to pass in calc_distance function
            return euclidean(sample, centroid)
    
        distances[i] = dataset.reduce_rows(calc_distance)

    # each dictionary entry is interpreted as a column
    return pd.DataFrame(distances)

def _compute_iteration(dataset, centroids):
    """
    Computes an iteration of the k-means algorithm.
    
    Args:
      dataset: model.DataSet
        The dataset being clustered.
      centroids: list of pandas Series
        The current centroids at the start of the iteration.
        
    Returns:
      new_centroids: list of pandas Series
        The updated centroids.
      cluster_assignments: pandas Series
        The current cluster assignments for each sample.
    """
    # 2. Calculate calc_distance from each data point to each centroid
    distances = _get_distances_to_centroids(dataset, centroids)

    # 3. Find each datapoint's nearest centroid
    cluster_assignments = distances.idxmin(axis=1)

    def nearest_centroid(sample_index):
        return cluster_assignments[sample_index]
        
    # 4. Calculate mean position of datapoints in each centroid's cluster
    new_centroids = dataset.get_data_frame().groupby(nearest_centroid).mean()

    # XXX turning each row in dataframe into a series... refactor!    
    list_of_series = [new_centroids.ix[ind] for ind in new_centroids.index]
    
    return list_of_series, cluster_assignments