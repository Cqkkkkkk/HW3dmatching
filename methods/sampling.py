import random
import numpy as np


def random_sampling(pts, num_samples):
    """
    Randomly samples a given number of points from a given set of points.

    Args:
        pts (array-like): The set of points to sample from.
        num_samples (int): The number of points to sample.

    Returns:
        array-like: An array of randomly sampled points.
    """
    return np.array(random.sample(list(pts), k=num_samples))


def voxel_grid_sampling(pts, voxel_size=0.001):
    """
    Perform voxel grid sampling on a point cloud.

    Args:
        pts (numpy.ndarray): The input point cloud, with shape (N, 3).
        voxel_size (float, optional): The size of each voxel. Defaults to 0.001.

    Returns:
        numpy.ndarray: The sampled point cloud, with shape (M, 3), where M <= N.
    """
    # Calculate the indices of each point in the grid
    indices = np.floor(pts / voxel_size).astype(np.int32)

    # Create a dictionary to hold points for each voxel
    voxels = {}
    for point, idx in zip(pts, indices):
        key = tuple(idx)
        if key in voxels:
            voxels[key].append(point)
        else:
            voxels[key] = [point]

    # Average the points in each voxel to create a sampled point cloud
    sampled = np.array([np.mean(points, axis=0) for points in voxels.values()])

    return sampled


def farthest_point_sampling(pts, num_samples):
    """
    Performs farthest point sampling on a set of points.

    Args:
        pts (numpy.ndarray): The input points.
        num_samples (int): The number of samples to select.

    Returns:
        numpy.ndarray: The selected samples.
    """
    num_nodes = pts.shape[0]
    num_samples = min(num_samples, num_nodes)
    indices = np.zeros(num_samples, dtype=int)
    distances = np.full(num_nodes, np.inf)
    farthest = np.random.randint(0, num_nodes)

    for i in range(num_samples):
        indices[i] = farthest
        centroid = pts[farthest, :]
        dist = np.sum((pts - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)

    return pts[indices, :]
