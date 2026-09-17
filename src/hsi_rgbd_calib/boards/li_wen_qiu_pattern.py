"""Li-Wen-Qiu calibration pattern model.

This module defines the planar pattern geometry used in the Li-Wen-Qiu
line-scan camera calibration method. The pattern consists of 6 feature
lines that are used for cross-ratio-based point correspondence.

Reference:
    Li, Wen, Qiu (2016). Cross-ratio-based line scan camera calibration
    using a planar pattern. Optical Engineering 55(1), 014104.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

import yaml

from hsi_rgbd_calib.common.logging import get_logger
from hsi_rgbd_calib.boards.geometry import (
    normalize_line,
    intersect_lines_2d,
    line_through_points,
)

logger = get_logger(__name__)


@dataclass
class LiWenQiuPattern:
    """Li-Wen-Qiu calibration pattern with 6 feature lines.
    
    The pattern lies in the Z=0 plane in pattern coordinates.
    Each feature line L_i is represented as (a, b, c) where ax + by + c = 0.
    
    Attributes:
        name: Pattern identifier.
        wp1: Width parameter 1 (meters) - used in cross-ratio formulas.
        wp2: Width parameter 2 (meters) - used in cross-ratio formulas.
        feature_lines: List of 6 line tuples [(a, b, c), ...].
        line_names: Names for each line ["L1", "L2", ...].
    """
    
    name: str
    wp1: float  # meters
    wp2: float  # meters
    feature_lines: List[Tuple[float, float, float]] = field(default_factory=list)
    line_names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate pattern configuration."""
        if len(self.feature_lines) != 6:
            raise ValueError(f"Pattern must have 6 feature lines, got {len(self.feature_lines)}")
        
        if len(self.line_names) == 0:
            self.line_names = [f"L{i+1}" for i in range(6)]
        
        # Normalize all lines
        self.feature_lines = [
            normalize_line(*line) for line in self.feature_lines
        ]
    
    def get_line(self, index: int) -> Tuple[float, float, float]:
        """Get feature line by index (0-5).
        
        Args:
            index: Line index (0 for L1, 1 for L2, etc.).
            
        Returns:
            Line coefficients (a, b, c).
        """
        if not 0 <= index < 6:
            raise IndexError(f"Line index {index} out of range [0, 5]")
        return self.feature_lines[index]
    
    def get_line_by_name(self, name: str) -> Tuple[float, float, float]:
        """Get feature line by name (e.g., "L1", "L2").
        
        Args:
            name: Line name.
            
        Returns:
            Line coefficients (a, b, c).
        """
        if name in self.line_names:
            return self.feature_lines[self.line_names.index(name)]
        raise KeyError(f"Unknown line name: {name}")
    
    def intersect_with_all_lines(
        self, 
        scan_line: Tuple[float, float, float]
    ) -> List[Optional[Tuple[float, float]]]:
        """Intersect a scan line with all 6 feature lines.
        
        Args:
            scan_line: Line coefficients (a, b, c) of the scan line.
            
        Returns:
            List of 6 intersection points (x, y), or None if parallel.
        """
        intersections = []
        for i, feature_line in enumerate(self.feature_lines):
            pt = intersect_lines_2d(scan_line, feature_line)
            intersections.append(pt)
        return intersections
    
    def compute_P3_P5_from_cross_ratios(
        self,
        x3: float,
        x5: float,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute P3 and P5 coordinates from X3 and X5.
        
        In the Li-Wen-Qiu pattern:
        - P3 lies on L3 (Y = wp2), so P3 = (X3, wp2, 0)
        - P5 lies on L5 (Y = wp1), so P5 = (X5, wp1, 0)
        
        Args:
            x3: X coordinate of P3 from cross-ratio.
            x5: X coordinate of P5 from cross-ratio.
            
        Returns:
            Tuple of P3=(x3, y3, 0) and P5=(x5, y5, 0) in pattern coords.
        """
        # P3 lies on L3: Y = wp2
        p3 = (x3, self.wp2, 0.0)
        
        # P5 lies on L5: Y = wp1
        p5 = (x5, self.wp1, 0.0)
        
        return p3, p5
    
    def compute_all_P_from_cross_ratios(
        self,
        x3: float,
        x5: float,
    ) -> List[Tuple[float, float, float]]:
        """Compute all 6 intersection points P1-P6 from cross-ratio results.
        
        Given X3 and X5 from cross-ratio formulas:
        1. Compute P3 and P5 on their respective feature lines
        2. Form the scan line through P3 and P5
        3. Intersect scan line with all 6 feature lines to get P1-P6
        
        Args:
            x3: X coordinate of P3.
            x5: X coordinate of P5.
            
        Returns:
            List of 6 points [(x, y, 0), ...] for P1 through P6.
        """
        p3, p5 = self.compute_P3_P5_from_cross_ratios(x3, x5)
        
        # Form scan line through P3 and P5 (in 2D, ignoring Z=0)
        scan_line = line_through_points((p3[0], p3[1]), (p5[0], p5[1]))
        
        # Intersect with all feature lines
        intersections = self.intersect_with_all_lines(scan_line)
        
        # Convert to 3D points (Z=0)
        points_3d = []
        for i, pt in enumerate(intersections):
            if pt is None:
                raise ValueError(f"Scan line parallel to feature line L{i+1}")
            points_3d.append((pt[0], pt[1], 0.0))
        
        return points_3d
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dimensions": {
                "wp1": self.wp1,
                "wp2": self.wp2,
            },
            "feature_lines": {
                name: list(line) for name, line in zip(self.line_names, self.feature_lines)
            },
        }


def load_li_wen_qiu_pattern(path: Path | str) -> LiWenQiuPattern:
    """Load Li-Wen-Qiu pattern from a YAML file.
    
    Args:
        path: Path to the pattern configuration file.
        
    Returns:
        Loaded LiWenQiuPattern.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If configuration is invalid.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Pattern file not found: {path}")
    
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    if data is None:
        raise ValueError(f"Empty pattern file: {path}")
    
    # Extract dimensions
    dims = data.get("dimensions", {})
    wp1 = float(dims.get("wp1", 0.1))
    wp2 = float(dims.get("wp2", 0.05))
    
    # Extract feature lines
    lines_data = data.get("feature_lines", {})
    
    feature_lines = []
    line_names = []
    
    for i in range(1, 7):
        name = f"L{i}"
        if name not in lines_data:
            raise ValueError(f"Missing feature line {name} in pattern file")
        
        line_coeffs = lines_data[name]
        if len(line_coeffs) != 3:
            raise ValueError(f"Line {name} must have 3 coefficients, got {len(line_coeffs)}")
        
        feature_lines.append(tuple(line_coeffs))
        line_names.append(name)
    
    return LiWenQiuPattern(
        name=data.get("name", "li_wen_qiu_pattern"),
        wp1=wp1,
        wp2=wp2,
        feature_lines=feature_lines,
        line_names=line_names,
    )


def get_default_li_wen_qiu_pattern() -> LiWenQiuPattern:
    """Get the default Li-Wen-Qiu right-triangle pattern.
    
    This pattern follows the conventions from Table 2 of the Li-Wen-Qiu paper:
    - L1, L2, L3: Horizontal lines (parallel to X-axis) at Y=0, Y=wp2, Y=wp1
    - L4, L5, L6: Diagonal lines (X=Y+c) with same slope, at offsets 0, wp2, wp1
    
    Note: The cross-ratio formulas in the paper are derived specifically for
    this pattern configuration. The point assignments are:
    - P1 on L1 (Y=0) and L4 (X=Y) -> P1 = (0, 0)
    - P2 on L2 (Y=wp2) and L4 (X=Y) -> P2 = (wp2, wp2)
    - P3 on L3 (Y=wp1) and L4 (X=Y) -> P3 = (wp1, wp1)
    - P4 on L1 (Y=0) and L5 (X=Y+wp2) -> P4 = (wp2, 0)
    - P5 on L1 (Y=0) and L6 (X=Y+wp1) -> P5 = (wp1, 0)
    - P6 on L2 (Y=wp2) and L6 (X=Y+wp1) -> P6 = (wp1+wp2, wp2)
    
    Returns:
        Default LiWenQiuPattern instance.
    """
    wp1 = 0.100  # 100mm
    wp2 = 0.050  # 50mm
    
    # Feature lines as (a, b, c) where ax + by + c = 0
    # Following paper's Table 2:
    # L1: Y = 0          -> 0*x + 1*y + 0 = 0
    # L2: Y = wp2        -> 0*x + 1*y - wp2 = 0
    # L3: Y = wp1        -> 0*x + 1*y - wp1 = 0
    # L4: X = Y          -> 1*x - 1*y + 0 = 0
    # L5: X = Y + wp2    -> 1*x - 1*y - wp2 = 0
    # L6: X = Y + wp1    -> 1*x - 1*y - wp1 = 0
    
    feature_lines = [
        (0.0, 1.0, 0.0),        # L1: Y = 0
        (0.0, 1.0, -wp2),       # L2: Y = wp2
        (0.0, 1.0, -wp1),       # L3: Y = wp1
        (1.0, -1.0, 0.0),       # L4: X = Y
        (1.0, -1.0, -wp2),      # L5: X - Y = wp2
        (1.0, -1.0, -wp1),      # L6: X - Y = wp1
    ]
    
    return LiWenQiuPattern(
        name="li_wen_qiu_default",
        wp1=wp1,
        wp2=wp2,
        feature_lines=feature_lines,
        line_names=["L1", "L2", "L3", "L4", "L5", "L6"],
    )

