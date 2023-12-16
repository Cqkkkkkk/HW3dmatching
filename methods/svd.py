import numpy as np
from methods.base import BasePointCloudProcessor


class SVDPointCloudProcessor(BasePointCloudProcessor):
	def __init__(self, sampling_mode='random'):
		self.sampling_mode = sampling_mode
		print(f'Now doing ICP with SVD method. Sampling mode: {sampling_mode}')
		super().__init__(save_dir='result/svd/', sampling_mode=sampling_mode)

	def icp_core_method(self, pts1, pts2, latest_transformation=None):
		pts1 = pts1[:, :3]
		pts2 = pts2[:, :3]
		row, col = pts1.shape
		weights = np.eye(row)
		p = pts2.T
		q = pts1.T
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
			rotation_matrix, pts2.T) + translation_matrix
		error = np.mean(
			np.sqrt(np.sum(np.square(registered_pts.T - pts1), axis=1)))
		new_transform = np.zeros((4, 4))
		new_transform[0:3, 0:3] = rotation_matrix
		new_transform[:3, 3] = translation_matrix.T
		new_transform[3, 3] = 1

		return new_transform, error
