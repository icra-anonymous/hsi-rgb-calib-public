"""2D geometry utilities for calibration.

This module provides robust 2D line geometry functions used
in the Li-Wen-Qiu calibration method.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray


def normalize_line(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """Normalize 2D line coefficients to have unit normal.
    
    Line equation: ax + by + c = 0
    After normalization: sqrt(a^2 + b^2) = 1
    
    Args:
        a, b, c: Line coefficients.
        
    Returns:
        Normalized coefficients (a', b', c').
        
    Raises:
        ValueError: If a=b=0 (degenerate line).
    """
    norm = np.sqrt(a * a + b * b)
    
    if norm < 1e-12:
        raise ValueError("Degenerate line: a and b are both near zero")
    
    return a / norm, b / norm, c / norm


def line_through_points(
    p1: Tuple[float, float], 
    p2: Tuple[float, float]
) -> Tuple[float, float, float]:
    """Compute 2D line through two points.
    
    Returns normalized line coefficients (a, b, c) where ax + by + c = 0.
    
    Args:
        p1: First point (x1, y1).
        p2: Second point (x2, y2).
        
    Returns:
        Normalized line coefficients (a, b, c).
        
    Raises:
        ValueError: If points are coincident.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Line: (y1 - y2)x + (x2 - x1)y + (x1*y2 - x2*y1) = 0
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    
    return normalize_line(a, b, c)


def intersect_lines_2d(
    l1: Tuple[float, float, float],
    l2: Tuple[float, float, float],
) -> Optional[Tuple[float, float]]:
    """Compute intersection of two 2D lines.
    
    Each line is represented as (a, b, c) where ax + by + c = 0.
    
    Args:
        l1: First line coefficients (a1, b1, c1).
        l2: Second line coefficients (a2, b2, c2).
        
    Returns:
        Intersection point (x, y), or None if lines are parallel.
    """
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    
    # Solve: a1*x + b1*y = -c1
    #        a2*x + b2*y = -c2
    det = a1 * b2 - a2 * b1
    
    if abs(det) < 1e-12:
        return None  # Lines are parallel
    
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    
    return (x, y)


def point_on_line(
    point: Tuple[float, float],
    line: Tuple[float, float, float],
    tol: float = 1e-6,
) -> bool:
    """Check if a point lies on a line.
    
    Args:
        point: Point (x, y).
        line: Line coefficients (a, b, c).
        tol: Tolerance for the check.
        
    Returns:
        True if point is on line within tolerance.
    """
    x, y = point
    a, b, c = line
    
    # For normalized line, residual is signed distance
    residual = abs(a * x + b * y + c)
    norm = np.sqrt(a * a + b * b)
    
    if norm < 1e-12:
        return False
    
    return residual / norm < tol


def signed_distance_to_line(
    point: Tuple[float, float],
    line: Tuple[float, float, float],
) -> float:
    """Compute signed distance from point to line.
    
    Args:
        point: Point (x, y).
        line: Line coefficients (a, b, c).
        
    Returns:
        Signed distance (positive on one side, negative on the other).
    """
    x, y = point
    a, b, c = line
    
    norm = np.sqrt(a * a + b * b)
    
    if norm < 1e-12:
        return float('inf')
    
    return (a * x + b * y + c) / norm


def cross_ratio_1d(v1: float, v2: float, v3: float, v4: float) -> float:
    """Compute cross-ratio of four collinear points.
    
    Cross-ratio CR(v1, v2, v3, v4) = (v1-v3)/(v2-v3) : (v1-v4)/(v2-v4)
                                   = ((v1-v3)*(v2-v4)) / ((v2-v3)*(v1-v4))
    
    Args:
        v1, v2, v3, v4: 1D coordinates of four collinear points.
        
    Returns:
        Cross-ratio value.
        
    Raises:
        ValueError: If denominator is zero (degenerate configuration).
    """
    num = (v1 - v3) * (v2 - v4)
    den = (v2 - v3) * (v1 - v4)
    
    if abs(den) < 1e-12:
        raise ValueError("Degenerate cross-ratio: denominator is zero")
    
    return num / den


def project_point_to_line(
    point: Tuple[float, float],
    line: Tuple[float, float, float],
) -> Tuple[float, float]:
    """Project a point onto a line.
    
    Args:
        point: Point (x, y).
        line: Line coefficients (a, b, c).
        
    Returns:
        Projected point (x', y').
    """
    x, y = point
    a, b, c = line
    
    # Normalize
    norm_sq = a * a + b * b
    if norm_sq < 1e-12:
        return point
    
    # Distance from point to line
    d = (a * x + b * y + c) / norm_sq
    
    # Projected point
    x_proj = x - a * d
    y_proj = y - b * d
    
    return (x_proj, y_proj)
