import numpy as np
from scipy.spatial.transform import Rotation


def read_asc(file_path):
    with open(file_path, mode="r") as file:
        lines = file.readlines()
        point_l = []
        lines = lines[2:]
        for line in lines:
            x, y, z = line.replace("\n", "").split(" ")
            x, y, z = float(x), float(y), float(z)
            point_l.append([x, y, z, 1])
        points = np.array(point_l)
    # print(f"total {points.shape[0]} number of points read from {file_path}")
    return points


def write_asc(points, file_path):
    with open(file_path, mode="w") as file:
        file.write("# Geomagic Studio\n")
        file.write("# New Model\n")
        points_num = points.shape[0]
        for p_idx in range(0, points_num):
            pos = points[p_idx]
            file.write(f"{pos[0]:.7f} {pos[1]:.7f} {pos[2]:.7f}\n")
    # print(f"total {points.shape[0]} number of points write to {file_path}")
    return True


def param2matrix(x):
    """
    Convert a parameter vector to a 3D transformation matrix.

    Args:
        x (numpy.ndarray): The parameter vector of shape (12,).

    Returns:
        numpy.ndarray: The transformation matrix of shape (4, 4).
    """
    transformation = np.zeros(shape=[4, 4])
    transformation[0:3, 0:4] = x.reshape([3, 4])
    transformation[3, 3] = 1
    return transformation


def matrix2param(transformation):
    """
    Converts a 3D transformation matrix to a parameter vector.

    Args:
        transformation (numpy.ndarray): The 3D transformation matrix.

    Returns:
        numpy.ndarray: The parameter vector obtained from the transformation matrix.
    """
    x = transformation[0:3, 0:4].reshape([-1])
    return x


def extract_rotation(transformation):
    """
    Extracts the rotation matrix from a transformation matrix.

    Parameters:
    transformation (numpy.ndarray): The transformation matrix.

    Returns:
    numpy.ndarray: The rotation matrix.
    """
    return transformation[0:3, 0:3]


def gen_loss_fn(args):
    pts1, pts2 = args

    def loss_fn(x):
        """
        Calculates the loss function for ICP.

        Args:
            x (numpy.ndarray): The parameter vector of shape (12,).

        Returns:
            float: The loss value.
        """
        fun_transformation = param2matrix(x)
        warp_pts2 = (fun_transformation @ pts2.T).T
        loss = np.sum((pts1 - warp_pts2) ** 2)
        return loss

    return loss_fn


def rotation_constraint(x):
    """
    Calculates the rotation constraint for ICP.

    Args:
        x (numpy.ndarray): The parameter vector of shape (12,).

    Returns:
        float: The constraint value.
    """
    transformation = param2matrix(x)
    rotation = extract_rotation(transformation)
    return np.sum(rotation @ rotation.T - np.eye(3)) ** 2


def gen_constraint():
    """
    Generate a constraint for optimization.

    Returns:
        dict: A dictionary representing the constraint.
    """
    constraint = ({
        "type": "eq",
        "fun": rotation_constraint
    })
    return constraint


def warp_pts(transformation, pts):
    """
    Applies a transformation matrix to a set of points.

    Args:
        transformation (numpy.ndarray): The transformation matrix.
        pts (numpy.ndarray): The points to be transformed.

    Returns:
        numpy.ndarray: The transformed points.
    """
    return (transformation @ pts.T).T


def quaternion_to_rotation_matrix(quaternion):
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quaternion (list): A list representing the quaternion [w, x, y, z].

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.

    """
    rotation_matrix = Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()

    return rotation_matrix
