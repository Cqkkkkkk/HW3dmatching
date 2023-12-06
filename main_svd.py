import os
import pdb
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from utils import warp_pts


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src.
    """
    distances = cdist(src, dst)
    indices = np.argmin(distances, axis=1)
    return dst[indices]

def best_fit_transform(A, B):
    """
    Calculate the best fit transform (rotation and translation) that maps A onto B.
    """
    assert len(A) == len(B)

    # Calculate centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute the covariance matrix
    H = np.dot(AA.T, BB)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = np.dot(Vt.T, U.T)

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    return R, t

def icp(A, B, max_iterations=20, tolerance=0.001):
    """
    Iterative Closest Point using SVD.
    """
    src = np.copy(A)
    for i in range(max_iterations):
        # Find nearest neighbors
        dst = nearest_neighbor(src, B)

        # Compute best fit transform
        R, t = best_fit_transform(src, dst)

        # Update the source
        src = np.dot(R, src.T).T + t

        # Check convergence
        mean_error = np.mean(np.linalg.norm(src - dst, axis=1))
        if mean_error < tolerance:
            break

    # Return aligned source and transformation
    return src, R, t

class PointCloudProcessor:
    def __init__(self):
        self.init_asc_path = f"data/C1.asc"
        self.init_asc = self.read_asc(self.init_asc_path)
        self.cur_asc = self.init_asc.copy()
        self.final_asc = self.cur_asc.copy()
        self.init_pcd_path = f"data/C1.asc"
        self.init_pcd = self.read_asc(self.init_pcd_path)
        self.test_pcd = self.init_pcd.copy()

    def process_point_clouds(self):
        for i in range(2, 11):
            print(f"ICP registering point_cloud:{1} and point_cloud:{i}")
            regis_asc_path = f"data/C{i}.asc"
            regis_asc = self.read_asc(regis_asc_path)
            regis_trans, _ = self.ICP_algorithm(
                self.final_asc.copy(),
                regis_asc.copy(),
                filter_thresh=20,
                name=f"icp_{i}"
            )
            warp_asc = warp_pts(regis_trans, pts=regis_asc)
            self.final_asc = np.concatenate([self.final_asc, warp_asc], axis=0)
        self.write_asc(self.final_asc, "result/final.asc")

        for i in range(2, 1):
            other_pcd_path = f"data/C{i}.asc"
            other_pcd = self.read_asc(other_pcd_path)
            self.test_pcd = np.concatenate([self.test_pcd, other_pcd], axis=0)
        self.write_asc(self.test_pcd, "result/init.asc")

    # Assuming these are your methods, if not, you can remove them
    def read_asc(self, file_path):
        with open(file_path, mode="r") as file:
            lines = file.readlines()
            point_L = []
            lines = lines[2:]
            for line in lines:
                x, y, z = line.replace("\n", "").split(" ")
                x, y, z = float(x), float(y), float(z)
                point_L.append([x, y, z, 1])
            points = np.array(point_L)
        # print(f"total {points.shape[0]} number of points read from {file_path}")
        return points

    def write_asc(self, points, file_path):
        with open(file_path, mode="w") as file:
            file.write("# Geomagic Studio\n")
            file.write("# New Model\n")
            points_num = points.shape[0]
            for p_idx in range(0, points_num):
                pos = points[p_idx]
                file.write(f"{pos[0]:.7f} {pos[1]:.7f} {pos[2]:.7f}\n")
        # print(f"total {points.shape[0]} number of points write to {file_path}")
        return True

    def ICP_algorithm(self, pts1, pts2, filter_thresh=1000000, tol=1e-7, max_iter=25, save_fig=True, name="default"):
        print("Solving ICP using quaternion-based approach")
        print(f"PC1: {pts1.shape} PC2: {pts2.shape}")


        sample_num = int(pts2.shape[0] // 100)
        sampled_pts2 = np.array(random.sample(list(pts2), k=sample_num))

        pts1 = pts1[:, :3]
        sampled_pts2 = sampled_pts2[:, :3]

        # Call to quaternion_based_icp
        transformed_pts1, final_error = quaternion_based_icp(
            pts1, 
            sampled_pts2, 
            max_iterations=max_iter, 
            tolerance=tol
        )

        return transformed_pts1, final_error


if __name__ == '__main__':
    processor = PointCloudProcessor()
    processor.process_point_clouds()
