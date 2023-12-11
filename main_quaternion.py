import os
import pdb
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from utils import param2matrix, matrix2param, gen_loss_fn, warp_pts, gen_constraint
from scipy.spatial.transform import Rotation as R


def quaternion_to_rotation_matrix(q):
    rotation_matrix = np.array([[np.square(q[0]) + np.square(q[1]) - np.square(q[2]) - np.square(q[3]),
                                 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
                                [2 * (q[1] * q[2] + q[0] * q[3]),
                                 np.square(q[0]) - np.square(q[1]) + np.square(q[2]) - np.square(q[3]),
                                 2 * (q[2] * q[3] - q[0] * q[1])],
                                [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]),
                                 np.square(q[0]) - np.square(q[1]) - np.square(q[2]) + np.square(q[3])]],
                               dtype=np.float32)
    return rotation_matrix

def quaternion_method(template_pts, register_pts):
        
        template_pts = template_pts[:, :3]
        register_pts = register_pts[:, :3]
        row, col = template_pts.shape
        mean_template_pts = np.mean(template_pts, axis=0)
        mean_register_pts = np.mean(register_pts, axis=0)
        cov = np.matmul((register_pts - mean_register_pts).T, (template_pts - mean_template_pts)) / row
        A = cov - cov.T
        delta = np.array([A[1, 2], A[2, 0], A[0, 1]], dtype=np.float32).T
        Q = np.zeros((4, 4), dtype=np.float32)
        Q[0, 0] = np.trace(cov)
        Q[0, 1:] = delta
        Q[1:, 0] = delta
        Q[1:, 1:] = cov + cov.T - np.trace(cov) * np.eye(3)
        lambdas, vecs = np.linalg.eig(Q)
        q = vecs[:, np.argmax(lambdas)]
        rotation_matrix = quaternion_to_rotation_matrix(q)
        translation_matrix = mean_template_pts - np.matmul(rotation_matrix, mean_register_pts)
        registered_points = np.matmul(rotation_matrix, register_pts.T).T + translation_matrix
        error = np.mean(np.sqrt(np.sum(np.square(registered_points - template_pts), axis=1)))

        new_transform = np.zeros((4, 4))
        new_transform[0:3, 0:3] = rotation_matrix
        new_transform[:3, 3] = translation_matrix.T
        new_transform[3, 3] = 1
        return new_transform, error

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
            regis_trans = self.ICP_algorithm(
                self.final_asc.copy(),
                regis_asc.copy(),
                filter_thresh=20,
                name=f"icp_{i}"
            )
            # pdb.set_trace()
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

    def find_correspondence(self, corres_pts1, corres_pts2, filter_thresh=1000000):
        neigh = NearestNeighbors(n_neighbors=1, radius=20)
        neigh.fit(corres_pts1)
        nbrs_dist, nbrs_idx = neigh.kneighbors(
            X=corres_pts2,
            n_neighbors=1,
            return_distance=True
        )
        dist_min, dist_argmin = nbrs_dist[:, 0], nbrs_idx[:, 0]
        dist_mask = np.where(dist_min <= filter_thresh, True, False)
        filtered_pts1_idx = dist_argmin[dist_mask].astype(np.int32)
        filtered_pts2_idx = np.where(dist_mask)[0].astype(np.int32)
        return corres_pts1[filtered_pts1_idx, :], corres_pts2[filtered_pts2_idx, :]

    def ICP_algorithm(self, pts1, pts2, filter_thresh=1000000, tol=1e-7, max_iter=25, save_fig=True, name="default"):
        print("Solving ICP using iterative algorithm")
        print(f"PC1: {pts1.shape} PC2: {pts2.shape}")
        loss_list = []
        trans_list = []
        trans_list.append(np.eye(N=4))

        # Samples a subset of pts2 and find the corresponding points in pts1
        # using the find_correspondence function
        sample_num = int(pts2.shape[0] // 100)
        pts2 = np.array(random.sample(list(pts2), k=sample_num))

        filtered_pts1, filtered_pts2 = self.find_correspondence(
            corres_pts1=pts1,
            corres_pts2=pts2,
            filter_thresh=filter_thresh
        )
        cur_pts2 = pts2

       
        for iter_idx in tqdm(range(0, max_iter)):
            # pdb.set_trace()
            new_transform, loss = quaternion_method(filtered_pts1, filtered_pts2)

            trans_list.append(new_transform)
            loss_list.append(loss)

            if loss < tol:
                break
            
            cur_pts2 = warp_pts(new_transform, pts2)
            # Adopt a nearest neighbor algorithm to find the closest points in pts1 for each point in pts2. 
            # It returns the indices of these points and a mask indicating which points in pts2 have 
            # a corresponding point in pts1 within the filter threshold.
            filtered_pts1, filtered_pts2 = self.find_correspondence(pts1, cur_pts2)

        # save fig
        if save_fig:
            save_path = f"result/{name}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            x = np.array(list(range(0, len(loss_list))))
            y = np.array(loss_list)
            plt.plot(x, y)
            plt.title(name)
            plt.savefig(save_path)
            plt.close()
        return trans_list[-1]


if __name__ == '__main__':
    processor = PointCloudProcessor()
    processor.process_point_clouds()
 