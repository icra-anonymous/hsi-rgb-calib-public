"""Coordinate frame definitions and conventions.

This module defines the coordinate frame conventions used throughout the
HSI-RGBD calibration toolkit.

Frame Naming Convention:
    Frames are named using lowercase identifiers. The transformation T_A_B
    transforms points from frame B to frame A.
    
    Example: T_oakrgb_hsi transforms points from HSI frame to OAK RGB frame.

Standard Frames:
    - rig: The rig reference frame (typically attached to the mounting plate)
    - oak_rgb: OAK-D RGB camera optical frame (Z forward, X right, Y down)
    - oak_depth: OAK-D depth camera optical frame
    - hsi: HSI line-scan camera slit frame
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class Frame(Enum):
    """Standard coordinate frame identifiers."""
    
    RIG = "rig"
    OAK_RGB = "oak_rgb"
    OAK_DEPTH = "oak_depth"
    HSI = "hsi"
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class FrameConvention:
    """Describes the coordinate convention for a frame.
    
    Attributes:
        name: Frame identifier.
        description: Human-readable description.
        x_axis: Description of X axis direction.
        y_axis: Description of Y axis direction.
        z_axis: Description of Z axis direction.
        notes: Additional notes about the frame.
    """
    
    name: str
    description: str
    x_axis: str
    y_axis: str
    z_axis: str
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "axes": {
                "x": self.x_axis,
                "y": self.y_axis,
                "z": self.z_axis,
            },
            "notes": self.notes,
        }


# Standard frame conventions
FRAME_CONVENTIONS: Dict[Frame, FrameConvention] = {
    Frame.RIG: FrameConvention(
        name="rig",
        description="Rig reference frame attached to mounting plate",
        x_axis="Right (when facing forward)",
        y_axis="Down",
        z_axis="Forward (direction of travel)",
        notes="Origin at center of mounting plate",
    ),
    
    Frame.OAK_RGB: FrameConvention(
        name="oak_rgb",
        description="OAK-D RGB camera optical frame",
        x_axis="Right (in image)",
        y_axis="Down (in image)",
        z_axis="Forward (optical axis, into scene)",
        notes="Standard camera coordinate convention. Origin at optical center.",
    ),
    
    Frame.OAK_DEPTH: FrameConvention(
        name="oak_depth",
        description="OAK-D depth camera optical frame",
        x_axis="Right (in image)",
        y_axis="Down (in image)",
        z_axis="Forward (optical axis, into scene)",
        notes="When aligned to RGB, coincides with oak_rgb frame.",
    ),
    
    Frame.HSI: FrameConvention(
        name="hsi",
        description="HSI line-scan camera slit frame",
        x_axis="Along slit (spatial dimension)",
        y_axis="Perpendicular to slit (not directly imaged)",
        z_axis="Forward (optical axis, into scene)",
        notes=(
            "For pushbroom cameras, only the X dimension is directly imaged. "
            "The Y dimension is acquired through platform motion."
        ),
    ),
}


def get_frame_convention(frame: Frame) -> FrameConvention:
    """Get the coordinate convention for a frame.
    
    Args:
        frame: Frame identifier.
        
    Returns:
        FrameConvention describing the coordinate system.
    """
    return FRAME_CONVENTIONS[frame]


def describe_transform(from_frame: Frame, to_frame: Frame) -> str:
    """Generate a human-readable description of a transform.
    
    Args:
        from_frame: Source frame.
        to_frame: Target frame.
        
    Returns:
        Description string.
    """
    return (
        f"T_{to_frame.value}_{from_frame.value}: "
        f"Transforms points from {from_frame.value} frame to {to_frame.value} frame"
    )
