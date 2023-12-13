import random
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from utils import warp_pts


class BasePointCloudProcessor:
    def __init__(self, save_dir='result/base/'):
        self.init_asc_path = f"data/C1.asc"
        self.init_asc = self.read_asc(self.init_asc_path)
        self.cur_asc = self.init_asc.copy()
        self.final_asc = self.cur_asc.copy()
        self.init_pcd_path = f"data/C1.asc"
        self.init_pcd = self.read_asc(self.init_pcd_path)
        self.test_pcd = self.init_pcd.copy()
        self.save_dir = save_dir

    def process_point_clouds(self):
        for i in range(2, 11):
            print(f"ICP registering point_cloud:{1} and point_cloud:{i}")
            regis_asc_path = f"data/C{i}.asc"
            regis_asc = self.read_asc(regis_asc_path)
            regis_trans = self.ICP(
                self.final_asc.copy(),
                regis_asc.copy(),
                filter_thresh=20,
            )
            # pdb.set_trace()
            warp_asc = warp_pts(regis_trans, pts=regis_asc)
            self.final_asc = np.concatenate([self.final_asc, warp_asc], axis=0)
        self.write_asc(self.final_asc, f"{self.save_dir}/final.asc")

        for i in range(2, 1):
            other_pcd_path = f"data/C{i}.asc"
            other_pcd = self.read_asc(other_pcd_path)
            self.test_pcd = np.concatenate([self.test_pcd, other_pcd], axis=0)
        self.write_asc(self.test_pcd,  f"{self.save_dir}/init.asc")
        print(f'Done! Results saved to {self.save_dir}')

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

    def sampling(self, pts, sample_num):
        return np.array(random.sample(list(pts), k=sample_num))