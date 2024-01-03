import numpy as np
from methods.base import BasePointCloudProcessor
from utils import quaternion_to_rotation_matrix


class QuaternionPointCloudProcessor(BasePointCloudProcessor):
	def __init__(self, sampling_mode='random'):
		self.sampling_mode = sampling_mode
		print(f'Now doing ICP with quaternion method. Sampling mode: {sampling_mode}')
		super().__init__(save_dir='result/quaternion/', sampling_mode=sampling_mode)

	def icp_core_method(self, pts1, pts2, latest_transformation=None):
		"""
		Perform the core method of the Iterative Closest Point (ICP) algorithm.

		Args:
			pts1 (numpy.ndarray): Array of 3D points representing the source point cloud.
			pts2 (numpy.ndarray): Array of 3D points representing the target point cloud.
			latest_transformation (numpy.ndarray, optional): Latest transformation matrix. Defaults to None.

		Returns:
			numpy.ndarray: The new transformation matrix.
			float: The alignment error.

		"""
		# Select only the first three columns (x, y, z) from pts1 and pts2
		pts1, pts2 = pts1[:, :3], pts2[:, :3]

		# Calculate the mean of the points
		mean_pts1 = np.mean(pts1, axis=0)
		mean_pts2 = np.mean(pts2, axis=0)

		# Compute the covariance matrix
		cov = np.matmul((pts2 - mean_pts2).T, (pts1 - mean_pts1)) / pts1.shape[0]

		# Compute the antisymmetric part of the covariance matrix
		a_matrix = cov - cov.T
		delta = np.array([a_matrix[1, 2], a_matrix[2, 0], a_matrix[0, 1]], dtype=np.float32)

		# Construct the Q matrix for quaternion calculation
		q_matrix = np.zeros((4, 4), dtype=np.float32)
		q_matrix[0, 0] = np.trace(cov)
		q_matrix[0, 1:] = delta
		q_matrix[1:, 0] = delta
		q_matrix[1:, 1:] = cov + cov.T - np.trace(cov) * np.eye(3)

		# Eigen decomposition to find the quaternion
		lambdas, eigen_vectors = np.linalg.eig(q_matrix)
		q = eigen_vectors[:, np.argmax(lambdas)]
		rotation_matrix = quaternion_to_rotation_matrix(q)

		# Compute the translation
		translation_matrix = mean_pts1 - np.matmul(rotation_matrix, mean_pts2)

		# Apply the transformation to pts2
		registered_points = np.matmul(rotation_matrix, pts2.T).T + translation_matrix

		# Compute the alignment error
		error = np.mean(np.sqrt(np.sum(np.square(registered_points - pts1), axis=1)))

		# Construct the new transformation matrix
		new_transform = np.zeros((4, 4), dtype=np.float32)
		new_transform[0:3, 0:3] = rotation_matrix
		new_transform[:3, 3] = translation_matrix
		new_transform[3, 3] = 1

		return new_transform, error
