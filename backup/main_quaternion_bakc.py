import os
import pdb
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R

from utils import warp_pts

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R


def quaternion_based_icp(A, B, max_iterations=50, tolerance=1e-6):
    def nearest_neighbor(src, dst):
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    def best_fit_transform(A, B):
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # Compute the covariance matrix
        H = np.dot(AA.T, BB)

        # Compute the optimal rotation using quaternion
        # Define the symmetric matrix N
        N = np.array([[H[0, 0] + H[1, 1] + H[2, 2], H[1, 2] - H[2, 1], H[2, 0] - H[0, 2], H[0, 1] - H[1, 0]],
                      [H[1, 2] - H[2, 1], H[0, 0] - H[1, 1] - H[2, 2], H[0, 1] + H[1, 0], H[2, 0] + H[0, 2]],
                      [H[2, 0] - H[0, 2], H[0, 1] + H[1, 0], H[1, 1] - H[0, 0] - H[2, 2], H[1, 2] + H[2, 1]],
                      [H[0, 1] - H[1, 0], H[2, 0] + H[0, 2], H[1, 2] + H[2, 1], H[2, 2] - H[0, 0] - H[1, 1]]])

        # Find the eigenvalues and eigenvectors of N
        eigenvalues, eigenvectors = np.linalg.eig(N)
        # Select the eigenvector corresponding to the maximum eigenvalue
        quaternion = eigenvectors[:, eigenvalues.argmax()]

        # Convert quaternion to rotation matrix
        rotation = R.from_quat(quaternion).as_matrix()

        # Compute the translation
        translation = centroid_B.T - np.dot(rotation, centroid_A.T)

        return quaternion, translation

    A = np.copy(A)
    for i in range(max_iterations):
        distances, indices = nearest_neighbor(A, B)

        quat, trans = best_fit_transform(A, B[indices])
        A = np.dot(R.from_quat(quat).as_matrix(), A.T).T + trans

        mean_error = np.mean(distances)
        if mean_error < tolerance:
            break

    final_rotation_matrix = R.from_quat(quat).as_matrix()
    final_transformation_matrix = np.eye(4)
    final_transformation_matrix[:3, :3] = final_rotation_matrix
    final_transformation_matrix[:3, 3] = trans

    return final_transformation_matrix, mean_error

# Example usage:
# A and B are 3xN matrices, where each column is a 3D point
# A_transformed, error = quaternion_based_icp(A, B)



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
            )
            warp_asc = warp_pts(regis_trans, pts=regis_asc)
            self.final_asc = np.concatenate([self.final_asc, warp_asc], axis=0)
        self.write_asc(self.final_asc, "result/quaternion/final.asc")

        for i in range(2, 1):
            other_pcd_path = f"data/C{i}.asc"
            other_pcd = self.read_asc(other_pcd_path)
            self.test_pcd = np.concatenate([self.test_pcd, other_pcd], axis=0)
        self.write_asc(self.test_pcd, "result/quaternion/init.asc")

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

    def ICP_algorithm(self, pts1, pts2, tol=1e-7, max_iter=25):
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
