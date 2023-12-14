import numpy as np
from scipy.spatial.transform import Rotation


def param2matrix(x):
    """
    Convert a parameter vector to a 3D transformation matrix.

    Args:
        x (numpy.ndarray): The parameter vector of shape (12,).

    Returns:
        numpy.ndarray: The transformation matrix of shape (4, 4).
    """
    T = np.zeros(shape=[4, 4])
    T[0:3, 0:4] = x.reshape([3, 4])
    T[3, 3] = 1
    return T


def matrix2param(T):
    """
    Converts a 3D transformation matrix to a parameter vector.

    Args:
        T (numpy.ndarray): The 3D transformation matrix.

    Returns:
        numpy.ndarray: The parameter vector obtained from the transformation matrix.
    """
    x = T[0:3, 0:4].reshape([-1])
    return x


def extrac_rotation(T):
    """
    Extracts the rotation matrix from a transformation matrix.

    Parameters:
    T (numpy.ndarray): The transformation matrix.

    Returns:
    numpy.ndarray: The rotation matrix.
    """
    return T[0:3, 0:3]

def gen_loss_fn(args):
    chosen_pts1, chosen_pts2 = args
    def loss_fn(x):
        """
        Calculates the loss function for ICP.

        Args:
            x (numpy.ndarray): The parameter vector of shape (12,).
            args (tuple): The arguments passed to the loss function.

        Returns:
            float: The loss value.
        """
        fun_T = param2matrix(x)
        warp_pts2 = (fun_T@(chosen_pts2.T)).T
        loss = np.sum((chosen_pts1 - warp_pts2)**2)
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
    T = param2matrix(x)
    R = extrac_rotation(T)
    return np.sum(R@R.T - np.eye(3))**2


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

def warp_pts(T, pts):
    """
    Applies a transformation matrix to a set of points.

    Args:
        T (numpy.ndarray): The transformation matrix.
        pts (numpy.ndarray): The points to be transformed.

    Returns:
        numpy.ndarray: The transformed points.
    """
    return (T@pts.T).T


def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion to a rotation matrix.

    Args:
        q (list): A list representing the quaternion [w, x, y, z].

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.

    """
    rotation_matrix = Rotation.from_quat(np.roll(q, -1)).as_matrix()
    
    return rotation_matrix