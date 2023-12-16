from methods.optimize import OptimizePointCloudProcessor
from methods.quaternion import QuaternionPointCloudProcessor
from methods.svd import SVDPointCloudProcessor

if __name__ == '__main__':
    processor = OptimizePointCloudProcessor(sampling_mode='voxel_grid')
    processor.process_point_clouds()
    processor = QuaternionPointCloudProcessor(sampling_mode='voxel_grid')
    processor.process_point_clouds()
    processor = SVDPointCloudProcessor(sampling_mode='voxel_grid')
    processor.process_point_clouds()
