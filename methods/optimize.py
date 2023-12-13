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
	def __init__(self):
		super().__init__(save_dir='result/optimize/')
		  
	def ICP(self, pts1, pts2, filter_thresh=1000000, max_iter=25):
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

		# Main function: A loop that iteratively finds the optimal transformation
		# In each iteration, it defines a loss function as well as contraints, then
		# use the minimize function to find the transformation that minimizes the loss
		for _ in tqdm(range(0, max_iter)):
			args = (pts1, cur_pts2, filtered_pts1, filtered_pts2)
			loss_fn = gen_loss_fn(args=args)
			x0 = matrix2param(trans_list[-1])
			constraints = gen_constraint()
			res = minimize(
				fun=loss_fn,
				x0=x0,
				method="SLSQP",
				constraints=constraints
			)
			trans_list.append(param2matrix(res.x))
			loss_list.append(res.fun)

			cur_pts2 = warp_pts(trans_list[-1], pts2)

			# Adopt a nearest neighbor algorithm to find the closest points in pts1 for each point in pts2. 
			# It returns the indices of these points and a mask indicating which points in pts2 have 
			# a corresponding point in pts1 within the filter threshold.
			filtered_pts1, filtered_pts2 = self.find_correspondence(
				corres_pts1=pts1,
				corres_pts2=cur_pts2,
				filter_thresh=filter_thresh
			)

		return trans_list[-1]


if __name__ == '__main__':
	processor = BasePointCloudProcessor()
	processor.process_point_clouds()
