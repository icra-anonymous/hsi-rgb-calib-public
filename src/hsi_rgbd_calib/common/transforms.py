"""SE(3) transformation utilities.

This module provides functions for working with rigid body transformations
represented as 4x4 homogeneous matrices.

Coordinate Convention:
    All transforms are represented as 4x4 homogeneous matrices in row-major order:
    
    T = [[R, t],
         [0, 1]]
    
    where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
    
    The transform T_A_B represents the transformation from frame B to frame A,
    such that a point p_B in frame B is transformed to frame A as:
    
        p_A = T_A_B @ p_B

Example:
    >>> import numpy as np
    >>> from hsi_rgbd_calib.common.transforms import compose, invert, apply_transform
    >>> 
    >>> # Create identity transform
    >>> T = np.eye(4)
    >>> 
    >>> # Apply to points
    >>> points = np.array([[1, 0, 0], [0, 1, 0]])
    >>> transformed = apply_transform(T, points)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def compose(T1: NDArray[np.float64], T2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compose two SE(3) transformations.
    
    Computes T1 @ T2, representing first applying T2, then T1.
    
    Args:
        T1: First 4x4 transformation matrix (applied second).
        T2: Second 4x4 transformation matrix (applied first).
        
    Returns:
        Composed 4x4 transformation matrix T1 @ T2.
        
    Raises:
        ValueError: If inputs are not 4x4 matrices.
        
    Example:
        >>> T_world_camera = compose(T_world_robot, T_robot_camera)
    """
    T1 = np.asarray(T1, dtype=np.float64)
    T2 = np.asarray(T2, dtype=np.float64)
    
    if T1.shape != (4, 4) or T2.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrices, got {T1.shape} and {T2.shape}")
    
    return T1 @ T2


def invert(T: NDArray[np.float64]) -> NDArray[np.float64]:
    """Invert an SE(3) transformation.
    
    For a rigid transformation T = [R, t; 0, 1], the inverse is:
    T^{-1} = [R^T, -R^T @ t; 0, 1]
    
    This is more numerically stable than np.linalg.inv for rigid transforms.
    
    Args:
        T: 4x4 transformation matrix.
        
    Returns:
        Inverted 4x4 transformation matrix.
        
    Raises:
        ValueError: If input is not a 4x4 matrix.
    """
    T = np.asarray(T, dtype=np.float64)
    
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {T.shape}")
    
    R = T[:3, :3]
    t = T[:3, 3]
    
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    
    return T_inv


def apply_transform(
    T: NDArray[np.float64], 
    points: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply an SE(3) transformation to a set of 3D points.
    
    Args:
        T: 4x4 transformation matrix.
        points: Nx3 array of 3D points.
        
    Returns:
        Nx3 array of transformed 3D points.
        
    Raises:
        ValueError: If inputs have incorrect shapes.
    """
    T = np.asarray(T, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {T.shape}")
    
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    if points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points.shape}")
    
    R = T[:3, :3]
    t = T[:3, 3]
    
    return (R @ points.T).T + t


def make_transform(
    R: NDArray[np.float64], 
    t: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Create a 4x4 homogeneous transformation matrix from R and t.
    
    Args:
        R: 3x3 rotation matrix.
        t: 3-element translation vector.
        
    Returns:
        4x4 transformation matrix.
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).flatten()
    
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got {R.shape}")
    if t.shape != (3,):
        raise ValueError(f"Expected 3-element translation, got {t.shape}")
    
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def decompose_transform(
    T: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Decompose a 4x4 transformation matrix into R and t.
    
    Args:
        T: 4x4 transformation matrix.
        
    Returns:
        Tuple of (R, t) where R is 3x3 and t is (3,).
    """
    T = np.asarray(T, dtype=np.float64)
    
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {T.shape}")
    
    R = T[:3, :3].copy()
    t = T[:3, 3].copy()
    
    return R, t


def rotation_matrix_to_euler(
    R: NDArray[np.float64], 
    order: str = "xyz"
) -> NDArray[np.float64]:
    """Convert a rotation matrix to Euler angles.
    
    Args:
        R: 3x3 rotation matrix.
        order: Euler angle order (e.g., "xyz", "zyx"). Default: "xyz".
        
    Returns:
        Array of 3 Euler angles in radians.
        
    Note:
        Uses the convention where rotations are applied in the order specified.
        For "xyz", rotate around X first, then Y, then Z.
    """
    from scipy.spatial.transform import Rotation
    
    R = np.asarray(R, dtype=np.float64)
    
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got {R.shape}")
    
    rot = Rotation.from_matrix(R)
    return rot.as_euler(order)


def euler_to_rotation_matrix(
    angles: NDArray[np.float64], 
    order: str = "xyz"
) -> NDArray[np.float64]:
    """Convert Euler angles to a rotation matrix.
    
    Args:
        angles: Array of 3 Euler angles in radians.
        order: Euler angle order (e.g., "xyz", "zyx"). Default: "xyz".
        
    Returns:
        3x3 rotation matrix.
    """
    from scipy.spatial.transform import Rotation
    
    angles = np.asarray(angles, dtype=np.float64).flatten()
    
    if angles.shape != (3,):
        raise ValueError(f"Expected 3 angles, got {angles.shape}")
    
    rot = Rotation.from_euler(order, angles)
    return rot.as_matrix()


def rotation_matrix_to_quaternion(
    R: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert a rotation matrix to a quaternion.
    
    Args:
        R: 3x3 rotation matrix.
        
    Returns:
        Quaternion as [w, x, y, z] (scalar-first convention).
    """
    from scipy.spatial.transform import Rotation
    
    R = np.asarray(R, dtype=np.float64)
    
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got {R.shape}")
    
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    
    # Convert to scalar-first [w, x, y, z]
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def quaternion_to_rotation_matrix(
    q: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert a quaternion to a rotation matrix.
    
    Args:
        q: Quaternion as [w, x, y, z] (scalar-first convention).
        
    Returns:
        3x3 rotation matrix.
    """
    from scipy.spatial.transform import Rotation
    
    q = np.asarray(q, dtype=np.float64).flatten()
    
    if q.shape != (4,):
        raise ValueError(f"Expected 4-element quaternion, got {q.shape}")
    
    # Convert from scalar-first [w, x, y, z] to scipy's [x, y, z, w]
    quat_scipy = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    
    rot = Rotation.from_quat(quat_scipy)
    return rot.as_matrix()


def is_valid_rotation_matrix(
    R: NDArray[np.float64], 
    tol: float = 1e-6
) -> bool:
    """Check if a matrix is a valid rotation matrix.
    
    A valid rotation matrix satisfies:
    - R @ R.T = I (orthogonality)
    - det(R) = 1 (proper rotation, not reflection)
    
    Args:
        R: Matrix to check.
        tol: Tolerance for numerical checks.
        
    Returns:
        True if R is a valid rotation matrix.
    """
    R = np.asarray(R, dtype=np.float64)
    
    if R.shape != (3, 3):
        return False
    
    # Check orthogonality
    should_be_identity = R @ R.T
    if not np.allclose(should_be_identity, np.eye(3), atol=tol):
        return False
    
    # Check determinant
    if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
        return False
    
    return True


def is_valid_transform(
    T: NDArray[np.float64], 
    tol: float = 1e-6
) -> bool:
    """Check if a matrix is a valid SE(3) transformation.
    
    Args:
        T: Matrix to check.
        tol: Tolerance for numerical checks.
        
    Returns:
        True if T is a valid SE(3) transformation.
    """
    T = np.asarray(T, dtype=np.float64)
    
    if T.shape != (4, 4):
        return False
    
    # Check bottom row
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=tol):
        return False
    
    # Check rotation part
    return is_valid_rotation_matrix(T[:3, :3], tol)
