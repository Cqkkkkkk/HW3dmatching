from scipy.optimize import minimize
from utils import param2matrix, matrix2param, gen_loss_fn, gen_constraint
from methods.base import BasePointCloudProcessor


class OptimizePointCloudProcessor(BasePointCloudProcessor):
	def __init__(self, sampling_mode='random'):
		self.sampling_mode = sampling_mode
		print(f'Now doing ICP with optimize method. Sampling mode: {sampling_mode}')
		super().__init__(save_dir='result/optimize/', sampling_mode=sampling_mode)

	def icp_core_method(self, pts1, pts2, latest_transformation):
		args = (pts1, pts2)
		loss_fn = gen_loss_fn(args=args)
		x0 = matrix2param(latest_transformation)
		constraints = gen_constraint()
		res = minimize(
			fun=loss_fn,
			x0=x0,
			method="SLSQP",
			constraints=constraints
		)
		return param2matrix(res.x), res.fun


if __name__ == '__main__':
	processor = BasePointCloudProcessor()
	processor.process_point_clouds()
