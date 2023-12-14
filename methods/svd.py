import os
import pdb
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from utils import param2matrix, matrix2param, gen_loss_fn, warp_pts, gen_constraint
from scipy.spatial.transform import Rotation as R
from methods.base import BasePointCloudProcessor


class SVDPointCloudProcessor(BasePointCloudProcessor):
	def __init__(self, sampling='random'):
		self.sampling = sampling
		super().__init__(save_dir='result/svd/')

	def svd_method(self, template_pts, register_pts):

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
		X = p - mean_p
		Y = q - mean_q
		S = np.matmul(np.matmul(X, weights), Y.T)
		U, sigma, VT = np.linalg.svd(S)
		det_V_Ut = np.linalg.det(np.matmul(VT.T, U.T))
		diag_matrix = np.eye(3)
		diag_matrix[2, 2] = det_V_Ut
		rotation_matrix = np.matmul(np.matmul(VT.T, diag_matrix), U.T)
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

	def ICP(self, pts1, pts2, filter_thresh=1000000, tol=1e-7, max_iter=25):
		loss_list = []
		trans_list = []
		trans_list.append(np.eye(N=4))

		# Samples a subset of pts2 and find the corresponding points in pts1
		# using the find_correspondence function
		if self.sampling == 'random':
			sample_num = int(pts2.shape[0] // 100)
			pts2 = self.random_sampling(pts2, sample_num=sample_num)
		elif self.sampling == 'voxel_grid':
			pts2 = self.voxel_grid_downsampling(pts2, voxel_size=0.001)
		else:
			raise NotImplementedError

		filtered_pts1, filtered_pts2 = self.find_correspondence(
			corres_pts1=pts1,
			corres_pts2=pts2,
			filter_thresh=filter_thresh
		)
		cur_pts2 = pts2

		for iter_idx in tqdm(range(0, max_iter)):
			# pdb.set_trace()
			new_transform, loss = self.svd_method(filtered_pts1, filtered_pts2)

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
