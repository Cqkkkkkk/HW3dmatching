import numpy as np
from tqdm import tqdm

from utils import warp_pts
from methods.base import BasePointCloudProcessor
from methods.sampling import random_sampling, voxel_grid_sampling, farthest_point_sampling


class SVDPointCloudProcessor(BasePointCloudProcessor):
	def __init__(self, sampling_mode='random'):
		self.sampling_mode = sampling_mode
		super().__init__(save_dir='result/svd/', save_name=sampling_mode)

	@staticmethod
	def svd_method(template_pts, register_pts):

		template_pts = template_pts[:, :3]
		register_pts = register_pts[:, :3]
		row, col = template_pts.shape
		weights = np.eye(row)
		p = register_pts.T
		q = template_pts.T
		mean_p = np.dot(p, np.diagonal(
			weights).reshape(-1, 1)) / np.trace(weights)
		mean_q = np.dot(q, np.diagonal(
			weights).reshape(-1, 1)) / np.trace(weights)
		x_matrix = p - mean_p
		y_matrix = q - mean_q
		s_matrix = np.matmul(np.matmul(x_matrix, weights), y_matrix.T)
		u_matrix, sigma, v_matrix = np.linalg.svd(s_matrix)
		det_v_ut = np.linalg.det(np.matmul(v_matrix.T, u_matrix.T))
		diag_matrix = np.eye(3)
		diag_matrix[2, 2] = det_v_ut
		rotation_matrix = np.matmul(np.matmul(v_matrix.T, diag_matrix), u_matrix.T)
		translation_matrix = mean_q - np.matmul(rotation_matrix, mean_p)
		registered_pts = np.matmul(
			rotation_matrix, register_pts.T) + translation_matrix
		error = np.mean(
			np.sqrt(np.sum(np.square(registered_pts.T - template_pts), axis=1)))
		new_transform = np.zeros((4, 4))
		new_transform[0:3, 0:3] = rotation_matrix
		new_transform[:3, 3] = translation_matrix.T
		new_transform[3, 3] = 1

		return new_transform, error

	def icp(self, pts1, pts2, filter_threshold=1000000, tol=1e-7, max_iter=25):
		loss_list = []
		trans_list = [np.eye(N=4)]

		# Samples a subset of pts2 and find the corresponding points in pts1
		pts2 = self.sampling(pts=pts2, mode=self.sampling_mode)
		print(f"Sampled pts2.shape: {pts2.shape}")

		filtered_pts1, filtered_pts2 = self.find_correspondence(
			pts1=pts1,
			pts2=pts2,
			filter_threshold=filter_threshold
		)

		for _ in tqdm(range(0, max_iter)):
			new_transform, loss = self.svd_method(filtered_pts1, filtered_pts2)

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
