import time
import tracemalloc
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from utils import warp_pts, read_asc, write_asc
from methods.sampling import random_sampling, voxel_grid_sampling, farthest_point_sampling


class BasePointCloudProcessor:
    """
    Base class for processing point clouds.

    Args:
        save_dir (str): Directory to save the results. Default is 'result/base/'.
        sampling_mode (str): Mode for sampling points. Default is 'random'.

    Attributes:
        init_asc_path (str): Path to the initial ASC file.
        init_asc (np.ndarray): Initial ASC data.
        cur_asc (np.ndarray): Current ASC data.
        final_asc (np.ndarray): Final ASC data.
        init_pcd_path (str): Path to the initial PCD file.
        init_pcd (np.ndarray): Initial PCD data.
        test_pcd (np.ndarray): Test PCD data.
        save_dir (str): Directory to save the results.
        save_name (str): Name of the saved file.
        sampling_mode (str): Mode for sampling points.

    Methods:
        process_point_clouds: Process the point clouds.
        find_correspondence: Find correspondences between two sets of points.
        icp_core_method: Core method for ICP (Iterative Closest Point) algorithm.
        sampling: Perform point sampling based on the specified mode.
        icp: Perform ICP registration between two point clouds.

    """

    def __init__(self, save_dir='result/base/', sampling_mode='random'):
        self.init_asc_path = f"data/C1.asc"
        self.init_asc = read_asc(self.init_asc_path)
        self.cur_asc = self.init_asc.copy()
        self.final_asc = self.cur_asc.copy()
        self.init_pcd_path = f"data/C1.asc"
        self.init_pcd = read_asc(self.init_pcd_path)
        self.test_pcd = self.init_pcd.copy()
        self.save_dir = save_dir
        self.save_name = sampling_mode
        self.sampling_mode = sampling_mode

    def process_point_clouds(self):
        """
        Process the point clouds by performing ICP registration and saving the results.
        """
        total_time = 0
        for i in range(2, 11):
            print(f"ICP registering point_cloud:{1} and point_cloud:{i}")
            regis_asc_path = f"data/C{i}.asc"
            regis_asc = read_asc(regis_asc_path)
            start_time = time.time()
            tracemalloc.start()
            regis_trans = self.icp(
                self.final_asc.copy(),
                regis_asc.copy(),
                filter_threshold=20,
            )
            end_time = time.time()
            print(f"ICP registering point_cloud:{1} and point_cloud:{i} takes {end_time - start_time} seconds")
            print(tracemalloc.get_traced_memory())
            total_time += end_time - start_time
            # pdb.set_trace()
            warp_asc = warp_pts(regis_trans, pts=regis_asc)
            self.final_asc = np.concatenate([self.final_asc, warp_asc], axis=0)
        print(f"ICP registering takes {total_time} seconds")
        write_asc(self.final_asc, f"{self.save_dir}/final_{self.save_name}.asc")

        for i in range(2, 1):
            other_pcd_path = f"data/C{i}.asc"
            other_pcd = read_asc(other_pcd_path)
            self.test_pcd = np.concatenate([self.test_pcd, other_pcd], axis=0)
        write_asc(self.test_pcd, f"{self.save_dir}/init_{self.save_name}.asc")
        print(f'Done! Results saved to {self.save_dir}')

    @staticmethod
    def find_correspondence(pts1, pts2, filter_threshold=1000000):
        """
        Find correspondences between two sets of points.

        Args:
            pts1 (np.ndarray): First set of points.
            pts2 (np.ndarray): Second set of points.
            filter_threshold (float): Distance threshold for filtering correspondences. Default is 1000000.

        Returns:
            np.ndarray: Filtered points from pts1.
            np.ndarray: Filtered points from pts2.
        """
        nearest_neighbor = NearestNeighbors(n_neighbors=1, radius=20)
        nearest_neighbor.fit(pts1)
        neighbor_distance, neighbor_index = nearest_neighbor.kneighbors(
            X=pts2,
            n_neighbors=1,
            return_distance=True
        )
        distance_min, distance_arg_min = neighbor_distance[:, 0], neighbor_index[:, 0]
        dist_mask = np.where(distance_min <= filter_threshold, True, False)
        filtered_pts1_idx = distance_arg_min[dist_mask].astype(np.int32)
        filtered_pts2_idx = np.where(dist_mask)[0].astype(np.int32)
        return pts1[filtered_pts1_idx, :], pts2[filtered_pts2_idx, :]

    def icp_core_method(self, pts1, pts2, latest_transformation):
        """
        Core method for the ICP (Iterative Closest Point) algorithm.

        Args:
            pts1 (np.ndarray): First set of points.
            pts2 (np.ndarray): Second set of points.
            latest_transformation (np.ndarray): Latest transformation matrix.

        Raises:
            NotImplementedError: This method should be implemented in the derived class.

        """
        raise NotImplementedError

    def sampling(self, pts, mode):
        """
        Perform point sampling based on the specified mode.

        Args:
            pts (np.ndarray): Set of points.
            mode (str): Sampling mode.

        Returns:
            np.ndarray: Sampled points.

        Raises:
            NotImplementedError: The specified sampling mode is not implemented.

        """
        if mode == 'random':
            num_samples = int(pts.shape[0] // 100)
            pts = random_sampling(pts, num_samples=num_samples)
        elif mode == 'voxel_grid':
            pts = voxel_grid_sampling(pts, voxel_size=0.001)
        elif mode == 'farthest':
            num_samples = int(pts.shape[0] // 100)
            pts = farthest_point_sampling(pts, num_samples=num_samples)
        else:
            raise NotImplementedError

        return pts

    def icp(self, pts1, pts2, filter_threshold=1000000, max_iter=25):
        """
        Perform ICP (Iterative Closest Point) registration between two point clouds.

        Args:
            pts1 (np.ndarray): First point cloud.
            pts2 (np.ndarray): Second point cloud.
            filter_threshold (float): Distance threshold for filtering correspondences. Default is 1000000.
            max_iter (int): Maximum number of iterations. Default is 25.

        Returns:
            np.ndarray: Transformation matrix.

        """
        loss_list = []
        trans_list = [np.eye(N=4)]
        pts2 = self.sampling(pts=pts2, mode=self.sampling_mode)
        print(f"Sampled pts2.shape: {pts2.shape}")

        filtered_pts1, filtered_pts2 = self.find_correspondence(
            pts1=pts1,
            pts2=pts2,
            filter_threshold=filter_threshold
        )

        for _ in tqdm(range(0, max_iter)):
            new_transform, loss = self.icp_core_method(
                pts1=filtered_pts1,
                pts2=filtered_pts2,
                latest_transformation=trans_list[-1]
            )

            trans_list.append(new_transform)
            loss_list.append(loss)

            cur_pts2 = warp_pts(new_transform, pts2)

            filtered_pts1, filtered_pts2 = self.find_correspondence(
                pts1=pts1,
                pts2=cur_pts2,
                filter_threshold=filter_threshold
            )

        return trans_list[-1]
