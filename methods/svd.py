import numpy as np
from methods.base import BasePointCloudProcessor


class SVDPointCloudProcessor(BasePointCloudProcessor):
	def __init__(self, sampling_mode='random'):
		self.sampling_mode = sampling_mode
		print(f'Now doing ICP with SVD method. Sampling mode: {sampling_mode}')
		super().__init__(save_dir='result/svd/', sampling_mode=sampling_mode)

	def icp_core_method(self, pts1, pts2, latest_transformation=None):
		"""
		Perform the Iterative Closest Point (ICP) core method to align two sets of 3D points.

		Args:
			pts1 (numpy.ndarray): Array of shape (N, 3) representing the first set of 3D points.
			pts2 (numpy.ndarray): Array of shape (N, 3) representing the second set of 3D points.
			latest_transformation (numpy.ndarray, optional): Transformation matrix from the previous iteration. Defaults to None.

		Returns:
			tuple: A tuple containing the new transformation matrix and the mean squared error.

		"""
		# Use only the x, y, z coordinates
		pts1, pts2 = pts1[:, :3], pts2[:, :3]

		# Rest of the code...
	def icp_core_method(self, pts1, pts2, latest_transformation=None):
		# Use only the x, y, z coordinates
		pts1, pts2 = pts1[:, :3], pts2[:, :3]

		# Number of points
		num_points = pts1.shape[0]

		# Initialize weights as an identity matrix
		weights = np.eye(num_points)

		# Transpose points for matrix operations
		p, q = pts2.T, pts1.T

		# Compute weighted means of points
		mean_p = np.dot(p, np.diagonal(weights).reshape(-1, 1)) / np.trace(weights)
		mean_q = np.dot(q, np.diagonal(weights).reshape(-1, 1)) / np.trace(weights)

		# Subtract means
		x_matrix = p - mean_p
		y_matrix = q - mean_q

		# Compute cross-covariance matrix
		s_matrix = np.matmul(np.matmul(x_matrix, weights), y_matrix.T)

		# Singular Value Decomposition
		u_matrix, sigma, v_matrix = np.linalg.svd(s_matrix)
		det_v_ut = np.linalg.det(np.matmul(v_matrix.T, u_matrix.T))

		# Ensure proper rotation (handling reflection)
		diag_matrix = np.eye(3)
		diag_matrix[2, 2] = det_v_ut

		# Calculate rotation matrix
		rotation_matrix = np.matmul(np.matmul(v_matrix.T, diag_matrix), u_matrix.T)

		# Calculate translation vector
		translation_matrix = mean_q - np.matmul(rotation_matrix, mean_p)

		# Transform pts2
		registered_pts = np.matmul(rotation_matrix, pts2.T) + translation_matrix

		# Calculate mean squared error
		error = np.mean(np.sqrt(np.sum(np.square(registered_pts.T - pts1), axis=1)))

		# Construct new transformation matrix
		new_transform = np.zeros((4, 4))
		new_transform[0:3, 0:3] = rotation_matrix
		new_transform[:3, 3] = translation_matrix.T
		new_transform[3, 3] = 1

		return new_transform, error
