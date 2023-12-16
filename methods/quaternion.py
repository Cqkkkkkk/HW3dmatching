import numpy as np
from methods.base import BasePointCloudProcessor
from utils import quaternion_to_rotation_matrix


class QuaternionPointCloudProcessor(BasePointCloudProcessor):
	def __init__(self, sampling_mode='random'):
		self.sampling_mode = sampling_mode
		print(f'Now doing ICP with quaternion method. Sampling mode: {sampling_mode}')
		super().__init__(save_dir='result/quaternion/', sampling_mode=sampling_mode)

	def icp_core_method(self, pts1, pts2, latest_transformation=None):
		pts1 = pts1[:, :3]
		pts2 = pts2[:, :3]
		row, col = pts1.shape
		mean_template_pts = np.mean(pts1, axis=0)
		mean_register_pts = np.mean(pts2, axis=0)
		cov = np.matmul((pts2 - mean_register_pts).T, (pts1 - mean_template_pts)) / row
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
			rotation_matrix, pts2.T).T + translation_matrix
		error = np.mean(
			np.sqrt(np.sum(np.square(registered_points - pts1), axis=1)))

		new_transform = np.zeros((4, 4))
		new_transform[0:3, 0:3] = rotation_matrix
		new_transform[:3, 3] = translation_matrix.T
		new_transform[3, 3] = 1
		return new_transform, error
