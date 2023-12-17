import numpy as np
from scipy.spatial.transform import Rotation


def read_asc(file_path):
    """
    Read an ASC file and return the points as a numpy array.

    Parameters:
    file_path (str): The path to the ASC file.

    Returns:
    numpy.ndarray: An array containing the x, y, z coordinates of the points.
    """
    with open(file_path, mode="r") as file:
        # Skip the first two lines and read the rest
        lines = file.readlines()[2:]
    # Process lines to extract x, y, z coordinates and convert them to floats
    point_l = [[float(coord) for coord in line.strip().split(" ")] + [1] for line in lines]
    points = np.array(point_l)
    return points


def write_asc(points, file_path):
    """
    Write the given points to a file in ASC format.

    Args:
        points (list): List of 3D points.
        file_path (str): Path to the output file.

    Returns:
        bool: True if the file was successfully written, False otherwise.
    """
    with open(file_path, mode="w") as file:
        file.write("# Geomagic Studio\n")
        file.write("# New Model\n")
        for pos in points:
            file.write(f"{pos[0]:.7f} {pos[1]:.7f} {pos[2]:.7f}\n")
    return True


def param2matrix(x):
    """
    Convert a 1D array of 12 elements into a 4x4 transformation matrix.

    Args:
        x (numpy.ndarray): Input array of shape (12,) containing the transformation parameters.

    Returns:
        numpy.ndarray: The resulting 4x4 transformation matrix.

    Raises:
        ValueError: If the input array does not have 12 elements.
    """
    if x.size != 12:
        raise ValueError("Input array must have 12 elements.")

    transformation = np.zeros((4, 4))
    transformation[0:3, 0:4] = x.reshape((3, 4))
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
    return transformation[0:3, 0:4].reshape([-1])


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
    """
    Generates a loss function for ICP.

    Args:
        args (tuple): A tuple containing two numpy arrays, pts1 and pts2.

    Returns:
        function: The loss function.

    """
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

    return Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()
