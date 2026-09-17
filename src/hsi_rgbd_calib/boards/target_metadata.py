"""Calibration target dimensions handling.

This module handles loading and managing physical dimensions
of calibration targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from hsi_rgbd_calib.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TargetDimensions:
    """Physical dimensions of a calibration target.
    
    Attributes:
        name: Target name/identifier.
        width_mm: Total width in millimeters.
        height_mm: Total height in millimeters.
        square_size_mm: Size of each square in millimeters.
        marker_size_mm: Size of ArUco markers in millimeters.
        border_mm: Border size in millimeters.
        print_scale: Scale factor for printing (1.0 = 100%).
        units: Unit system used ("mm" or "m").
        notes: Additional notes about the target.
    """
    
    name: str
    width_mm: float
    height_mm: float
    square_size_mm: float
    marker_size_mm: float
    border_mm: float = 10.0
    print_scale: float = 1.0
    units: str = "mm"
    notes: str = ""
    
    @property
    def square_size_m(self) -> float:
        """Get square size in meters."""
        return self.square_size_mm / 1000.0
    
    @property
    def marker_size_m(self) -> float:
        """Get marker size in meters."""
        return self.marker_size_mm / 1000.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "square_size_mm": self.square_size_mm,
            "marker_size_mm": self.marker_size_mm,
            "border_mm": self.border_mm,
            "print_scale": self.print_scale,
            "units": self.units,
            "notes": self.notes,
        }


def load_target_dimensions(path: Path | str) -> TargetDimensions:
    """Load target dimensions from a YAML file.
    
    Args:
        path: Path to the dimensions file.
        
    Returns:
        Loaded TargetDimensions.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dimensions file not found: {path}")
    
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    if data is None:
        raise ValueError(f"Empty dimensions file: {path}")
    
    # Handle nested structure (dimensions.yaml may have a top-level 'target' key)
    if "target" in data:
        data = data["target"]
    
    required_fields = ["name", "width_mm", "height_mm", "square_size_mm", "marker_size_mm"]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in dimensions file")
    
    return TargetDimensions(
        name=data["name"],
        width_mm=float(data["width_mm"]),
        height_mm=float(data["height_mm"]),
        square_size_mm=float(data["square_size_mm"]),
        marker_size_mm=float(data["marker_size_mm"]),
        border_mm=float(data.get("border_mm", 10.0)),
        print_scale=float(data.get("print_scale", 1.0)),
        units=data.get("units", "mm"),
        notes=data.get("notes", ""),
    )


def get_a4_charuco_dimensions() -> TargetDimensions:
    """Get default A4-sized ChArUco target dimensions.
    
    Returns:
        TargetDimensions for a 6x9 ChArUco board on A4 paper.
    """
    return TargetDimensions(
        name="charuco_6x9_a4",
        width_mm=210.0,  # A4 width
        height_mm=297.0,  # A4 height
        square_size_mm=30.0,
        marker_size_mm=22.5,  # 75% of square size
        border_mm=15.0,
        print_scale=1.0,
        units="mm",
        notes="Standard 6x9 ChArUco board for A4 paper",
    )
