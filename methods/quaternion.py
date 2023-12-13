import pdb
import random
import numpy as np
from tqdm import tqdm
from methods.base import BasePointCloudProcessor
from utils import quaternion_to_rotation_matrix, warp_pts


class QuaternionPointCloudProcessor(BasePointCloudProcessor):
    def __init__(self):
        super().__init__(save_dir='result/quaternion/')

    def quaternion_method(self, template_pts, register_pts):

        template_pts = template_pts[:, :3]
        register_pts = register_pts[:, :3]
        row, col = template_pts.shape
        mean_template_pts = np.mean(template_pts, axis=0)
        mean_register_pts = np.mean(register_pts, axis=0)
        cov = np.matmul((register_pts - mean_register_pts).T,
                        (template_pts - mean_template_pts)) / row
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
        translation_matrix = mean_template_pts - \
            np.matmul(rotation_matrix, mean_register_pts)
        registered_points = np.matmul(
            rotation_matrix, register_pts.T).T + translation_matrix
        error = np.mean(
            np.sqrt(np.sum(np.square(registered_points - template_pts), axis=1)))

        new_transform = np.zeros((4, 4))
        new_transform[0:3, 0:3] = rotation_matrix
        new_transform[:3, 3] = translation_matrix.T
        new_transform[3, 3] = 1
        return new_transform, error

    def ICP(self, pts1, pts2, filter_thresh=1000000, tol=1e-7, max_iter=25):
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

        for _ in tqdm(range(0, max_iter)):

            new_transform, loss = self.quaternion_method(
                filtered_pts1, filtered_pts2)

            trans_list.append(new_transform)
            loss_list.append(loss)

            if loss < tol:
                break

            cur_pts2 = warp_pts(new_transform, pts2)

            # Adopt a nearest neighbor algorithm to find the closest points in pts1 for each point in pts2.
            # It returns the indices of these points and a mask indicating which points in pts2 have
            # a corresponding point in pts1 within the filter threshold.
            filtered_pts1, filtered_pts2 = self.find_correspondence(
                pts1, cur_pts2)
            
        return trans_list[-1]
