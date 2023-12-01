import os
import pdb
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from utils import param2matrix, matrix2param, gen_loss_fn, warp_pts, gen_constraint


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
        corres_pmt1 = dist_argmin[dist_mask].astype(np.int32)
        corres_pmt2 = np.where(dist_mask)[0].astype(np.int32)
        return corres_pmt1, corres_pmt2, dist_mask

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
        cur_pmt1, cur_pmt2, _ = self.find_correspondence(
            corres_pts1=pts1,
            corres_pts2=pts2,
            filter_thresh=filter_thresh
        )
        cur_pts2 = pts2

        # Main function: A loop that iteratively finds the optimal transformation
        # In each iteration, it defines a loss function as well as contraints, then
        # use the minimize function to find the transformation that minimizes the loss
        for iter_idx in tqdm(range(0, max_iter)):
            args = (pts1, cur_pts2, cur_pmt1, cur_pmt2)
            loss_fn = gen_loss_fn(args=args)
            x0 = matrix2param(trans_list[-1])
            constraints = gen_constraint()
            res = minimize(
                fun=loss_fn,
                x0=x0,
                method="SLSQP",
                constraints=constraints
            )
            pdb.set_trace()
            trans_list.append(param2matrix(res.x))
            loss_list.append(res.fun)
            print(f"iter time:{iter_idx} x:{res.x} fun:{res.fun}")
            # print(f"transform:{trans_list[-1]}")


            cur_pts2 = warp_pts(trans_list[-1], pts2)

            # Adopt a nearest neighbor algorithm to find the closest points in pts1 for each point in pts2. 
            # It returns the indices of these points and a mask indicating which points in pts2 have 
            # a corresponding point in pts1 within the filter threshold.
            cur_pmt1, cur_pmt2, _ = self.find_correspondence(
                corres_pts1=pts1,
                corres_pts2=cur_pts2,
                filter_thresh=filter_thresh
            )

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
