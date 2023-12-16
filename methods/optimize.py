import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from utils import param2matrix, matrix2param, gen_loss_fn, warp_pts, gen_constraint
from methods.base import BasePointCloudProcessor
from methods.sampling import random_sampling, voxel_grid_sampling, farthest_point_sampling


class OptimizePointCloudProcessor(BasePointCloudProcessor):
	def __init__(self, sampling_mode='random'):
		self.sampling_mode = sampling_mode
		super().__init__(save_dir='result/optimize/', save_name=sampling_mode)

	@staticmethod
	def optimize_method(pts1, pts2, trans_list):
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

	def icp(self, pts1, pts2, filter_threshold=1000000, max_iter=25):
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
			new_transform, loss = self.optimize_method(
				pts1=filtered_pts1,
				pts2=filtered_pts2,
				trans_list=trans_list
			)

			trans_list.append(new_transform)
			loss_list.append(loss)

			cur_pts2 = warp_pts(new_transform, pts2)

			filtered_pts1, filtered_pts2 = self.find_correspondence(
				pts1=pts1,
				pts2=cur_pts2,
				filter_threshold=filter_threshold
			)

		return trans_list[-1]


if __name__ == '__main__':
	processor = BasePointCloudProcessor()
	processor.process_point_clouds()
