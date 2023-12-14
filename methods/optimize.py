import os
import pdb
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from utils import param2matrix, matrix2param, gen_loss_fn, warp_pts, gen_constraint
from methods.base import BasePointCloudProcessor


class OptimizePointCloudProcessor(BasePointCloudProcessor):
	def __init__(self, sampling='random'):
		self.sampling = sampling
		super().__init__(save_dir='result/optimize/', save_name=sampling)
	
	def optimize_method(self, pts1, pts2, trans_list):
		args = (pts1, pts2)
		loss_fn = gen_loss_fn(args=args)
		x0 = matrix2param(trans_list[-1])
		constraints = gen_constraint()
		res = minimize(
			fun=loss_fn,
			x0=x0,
			method="SLSQP",
			constraints=constraints
		)
		return param2matrix(res.x), res.fun
	

	def ICP(self, pts1, pts2, filter_thresh=1000000, max_iter=25):
		loss_list = []
		trans_list = []
		trans_list.append(np.eye(N=4))

		if self.sampling == 'random':
			sample_num = int(pts2.shape[0] // 100)
			pts2 = self.random_sampling(pts2, sample_num=sample_num)
		elif self.sampling == 'voxel_grid':
			pts2 = self.voxel_grid_downsampling(pts2, voxel_size=0.001)
		else:
			raise NotImplementedError
		print(f"Sampled pts2.shape: {pts2.shape}")
		
		filtered_pts1, filtered_pts2 = self.find_correspondence(
			corres_pts1=pts1,
			corres_pts2=pts2,
			filter_thresh=filter_thresh
		)
		cur_pts2 = pts2


		for _ in tqdm(range(0, max_iter)):

			new_transform, loss = self.optimize_method(
				pts1=filtered_pts1, 
				pts2=filtered_pts2, 
				trans_list=trans_list
			)
			
			trans_list.append(new_transform)
			loss_list.append(loss)

			cur_pts2 = warp_pts(new_transform, pts2)

			filtered_pts1, filtered_pts2 = self.find_correspondence(
				corres_pts1=pts1,
				corres_pts2=cur_pts2,
				filter_thresh=filter_thresh
			)

		return trans_list[-1]


if __name__ == '__main__':
	processor = BasePointCloudProcessor()
	processor.process_point_clouds()
