from methods.optimize import OptimizePointCloudProcessor
from methods.quaternion import QuaternionPointCloudProcessor
from methods.svd import SVDPointCloudProcessor

if __name__ == '__main__':
    # processor = OptimizePointCloudProcessor()
    # processor.process_point_clouds()
    processor = QuaternionPointCloudProcessor(sampling='voxel_grid')
    processor.process_point_clouds()
    processor = SVDPointCloudProcessor()
    processor.process_point_clouds()
