"""Cross-ratio computation for Li-Wen-Qiu calibration.

This module implements point correspondence establishment using cross-ratio invariance.

The pattern has 6 feature lines:
- L1: Y = 0 (horizontal)
- L2: Y = wp2 (horizontal)
- L3: Y = wp1 (horizontal)
- L4: X = Y (diagonal)
- L5: X - Y = wp2 (diagonal)
- L6: X - Y = wp1 (diagonal)

Cross-ratio invariance: CR(Pa, Pb, Pc, Pd) on pattern = CR(va, vb, vc, vd) observed

Using CR(P1, P2, PN, P3) where P1, P2, P3 are on known horizontal lines (Y=0, wp2, wp1),
we can recover the Y-coordinate of any diagonal point PN:

    yN = (CR * wp1 * wp2) / (wp2 + wp1 * (CR - 1))

Then X-coordinates are computed from the line equations:
- For L4 (X=Y): X4 = Y4
- For L5 (X-Y=wp2): X5 = Y5 + wp2
- For L6 (X-Y=wp1): X6 = Y6 + wp1
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray


def compute_cross_ratio(va: float, vb: float, vc: float, vd: float) -> float:
    """Compute cross-ratio CR(a, b; c, d) = ((a-c)*(b-d)) / ((b-c)*(a-d)).
    
    Args:
        va, vb, vc, vd: Four collinear point coordinates.
        
    Returns:
        Cross-ratio value.
        
    Raises:
        ValueError: If denominator is near zero.
    """
    den = (vb - vc) * (va - vd)
    if abs(den) < 1e-12:
        raise ValueError("Degenerate cross-ratio: denominator near zero")
    
    return ((va - vc) * (vb - vd)) / den


def recover_y_from_cross_ratio(
    cr: float,
    wp1: float, 
    wp2: float,
) -> float:
    """Recover Y-coordinate of a diagonal point from cross-ratio.
    
    Given CR(P1, P2, PN, P3) where:
    - P1 is on L1 (Y=0)
    - P2 is on L2 (Y=wp2)  
    - P3 is on L3 (Y=wp1)
    - PN is on a diagonal line
    
    The formula is: yN = (CR * wp1 * wp2) / (wp2 + wp1 * (CR - 1))
    
    Args:
        cr: Cross-ratio CR(v1, v2, vN, v3).
        wp1: Pattern parameter (Y-coord of L3).
        wp2: Pattern parameter (Y-coord of L2).
        
    Returns:
        Y-coordinate of the diagonal point.
    """
    den = wp2 + wp1 * (cr - 1.0)
    if abs(den) < 1e-12:
        raise ValueError(f"Degenerate Y recovery: denominator = {den}")
    
    return (cr * wp1 * wp2) / den


def recover_pattern_points_from_observations(
    v_obs: List[float],
    wp1: float,
    wp2: float,
    pattern_lines: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """Recover 3D pattern points P1-P6 from observed line-scan coordinates.
    
    Uses cross-ratio invariance to determine the position of points on
    diagonal feature lines (L4, L5, L6), then computes all 6 intersection
    points with the scan line.
    
    Algorithm:
    1. Compute CR(v1, v2, v4, v3) to find Y4, then X4 = Y4 (since L4: X=Y)
    2. Compute CR(v1, v2, v5, v3) to find Y5, then X5 = Y5 + wp2 (L5: X-Y=wp2)
    3. Compute CR(v1, v2, v6, v3) to find Y6, then X6 = Y6 + wp1 (L6: X-Y=wp1)
    4. Form scan line through P4, P5, P6
    5. Intersect scan line with L1, L2, L3 to get P1, P2, P3
    
    Args:
        v_obs: List of 6 observed pixel coordinates [v1, v2, v3, v4, v5, v6].
        wp1: Pattern width parameter 1 (Y-coord of L3).
        wp2: Pattern width parameter 2 (Y-coord of L2).
        pattern_lines: List of 6 feature line tuples [(a, b, c), ...].
        
    Returns:
        List of 6 3D points [(x, y, 0), ...] in pattern coordinates.
    """
    from hsi_rgbd_calib.boards.geometry import (
        intersect_lines_2d,
        line_through_points,
    )
    
    if len(v_obs) != 6:
        raise ValueError(f"Expected 6 observations, got {len(v_obs)}")
    
    v1, v2, v3, v4, v5, v6 = v_obs
    
    # Step 1: Recover Y4, Y5, Y6 using cross-ratios
    # CR(P1, P2, P4, P3) -> Y4
    cr4 = compute_cross_ratio(v1, v2, v4, v3)
    y4 = recover_y_from_cross_ratio(cr4, wp1, wp2)
    x4 = y4  # L4: X = Y
    
    # CR(P1, P2, P5, P3) -> Y5
    cr5 = compute_cross_ratio(v1, v2, v5, v3)
    y5 = recover_y_from_cross_ratio(cr5, wp1, wp2)
    x5 = y5 + wp2  # L5: X - Y = wp2
    
    # CR(P1, P2, P6, P3) -> Y6
    cr6 = compute_cross_ratio(v1, v2, v6, v3)
    y6 = recover_y_from_cross_ratio(cr6, wp1, wp2)
    x6 = y6 + wp1  # L6: X - Y = wp1
    
    # Step 2: Form scan line through P4 and P5 (or P5 and P6)
    p4_2d = (x4, y4)
    p5_2d = (x5, y5)
    
    scan_line = line_through_points(p4_2d, p5_2d)
    
    # Step 3: Intersect scan line with all feature lines to get all points
    points_3d = []
    for i, feature_line in enumerate(pattern_lines):
        pt = intersect_lines_2d(scan_line, feature_line)
        if pt is None:
            raise ValueError(f"Scan line parallel to feature line L{i+1}")
        points_3d.append((pt[0], pt[1], 0.0))
    
    return points_3d


# Keep old API for backward compatibility
def compute_cross_ratios(
    v1: float, v2: float, v3: float, v4: float, v5: float, v6: float
) -> Tuple[float, float]:
    """Compute cross-ratios CR(v1,v2,v4,v3) and CR(v1,v2,v5,v3).
    
    This is kept for backward compatibility. New code should use
    compute_cross_ratio() directly.
    """
    cr4 = compute_cross_ratio(v1, v2, v4, v3)
    cr5 = compute_cross_ratio(v1, v2, v5, v3)
    return cr4, cr5


def compute_X3_X5_from_cross_ratios(
    CR1: float,
    CR2: float,
    wp1: float,
    wp2: float,
) -> Tuple[float, float]:
    """Compute X coordinates from cross-ratios.
    
    This function is kept for backward compatibility. The naming refers to
    the OLD pattern indexing. In the new indexing:
    - CR1 gives X4 (point on L4)
    - CR2 gives X5 (point on L5)
    """
    y4 = recover_y_from_cross_ratio(CR1, wp1, wp2)
    y5 = recover_y_from_cross_ratio(CR2, wp1, wp2)
    x4 = y4  # L4: X = Y
    x5 = y5 + wp2  # L5: X - Y = wp2
    return x4, x5


def validate_cross_ratio_ordering(v_obs: List[float]) -> bool:
    """Validate that observations have reasonable values.
    
    Args:
        v_obs: List of 6 observed coordinates.
        
    Returns:
        True if values are valid.
    """
    if len(v_obs) != 6:
        return False
    
    v = np.array(v_obs)
    
    # Check uniqueness
    if len(np.unique(v)) != 6:
        return False
    
    # Check for NaN or inf
    if np.any(~np.isfinite(v)):
        return False
    
    return True
