import numpy as np
from tqdm import tqdm
from methods.base import BasePointCloudProcessor
from utils import quaternion_to_rotation_matrix, warp_pts
from methods.sampling import random_sampling, voxel_grid_sampling, farthest_point_sampling


class QuaternionPointCloudProcessor(BasePointCloudProcessor):
	def __init__(self, sampling_mode='random'):
		self.sampling_mode = sampling_mode
		super().__init__(save_dir='result/quaternion/', save_name=sampling_mode)

	@staticmethod
	def quaternion_method(template_pts, register_pts):
		template_pts = template_pts[:, :3]
		register_pts = register_pts[:, :3]
		row, col = template_pts.shape
		mean_template_pts = np.mean(template_pts, axis=0)
		mean_register_pts = np.mean(register_pts, axis=0)
		cov = np.matmul((register_pts - mean_register_pts).T, (template_pts - mean_template_pts)) / row
		a_matrix = cov - cov.T
		delta = np.array([a_matrix[1, 2], a_matrix[2, 0], a_matrix[0, 1]], dtype=np.float32).T
		q_matrix = np.zeros((4, 4), dtype=np.float32)
		q_matrix[0, 0] = np.trace(cov)
		q_matrix[0, 1:] = delta
		q_matrix[1:, 0] = delta
		q_matrix[1:, 1:] = cov + cov.T - np.trace(cov) * np.eye(3)
		lambdas, eigen_vectors = np.linalg.eig(q_matrix)
		q = eigen_vectors[:, np.argmax(lambdas)]
		rotation_matrix = quaternion_to_rotation_matrix(q)
		translation_matrix = mean_template_pts - np.matmul(rotation_matrix, mean_register_pts)
		registered_points = np.matmul(
			rotation_matrix, register_pts.T).T + translation_matrix
		error = np.mean(
			np.sqrt(np.sum(np.square(registered_points - template_pts), axis=1)))

		new_transform = np.zeros((4, 4))
		new_transform[0:3, 0:3] = rotation_matrix
		new_transform[:3, 3] = translation_matrix.T
		new_transform[3, 3] = 1
		return new_transform, error

	def icp(self, pts1, pts2, filter_threshold=1000000, tol=1e-7, max_iter=25):
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

			new_transform, loss = self.quaternion_method(
				filtered_pts1, filtered_pts2)

			trans_list.append(new_transform)
			loss_list.append(loss)

			if loss < tol:
				break

			cur_pts2 = warp_pts(new_transform, pts2)

			filtered_pts1, filtered_pts2 = self.find_correspondence(
				pts1=pts1,
				pts2=cur_pts2,
				filter_threshold=filter_threshold
			)

		return trans_list[-1]
