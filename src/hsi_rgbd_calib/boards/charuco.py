"""ChArUco board utilities.

This module provides utilities for working with ChArUco calibration boards,
including loading board configurations, creating board instances, and
detecting corners in images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import yaml

try:
    import cv2
    from cv2 import aruco
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from hsi_rgbd_calib.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BoardConfig:
    """Configuration for a ChArUco calibration board.
    
    Attributes:
        name: Board name/identifier.
        board_type: Type of board ("charuco", "apriltag").
        squares_x: Number of chessboard squares in X direction.
        squares_y: Number of chessboard squares in Y direction.
        square_length_m: Size of chessboard square in meters.
        marker_length_m: Size of ArUco marker in meters.
        dictionary: ArUco dictionary name (e.g., "DICT_6X6_250").
        border_bits: Number of marker border bits.
    """
    
    name: str
    board_type: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    dictionary: str = "DICT_6X6_250"
    border_bits: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "board_type": self.board_type,
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_m": self.square_length_m,
            "marker_length_m": self.marker_length_m,
            "dictionary": self.dictionary,
            "border_bits": self.border_bits,
        }


def load_board_config(path: Path | str) -> BoardConfig:
    """Load board configuration from a YAML file.
    
    Args:
        path: Path to the configuration file.
        
    Returns:
        Loaded BoardConfig.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Board config file not found: {path}")
    
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    if data is None:
        raise ValueError(f"Empty board config file: {path}")
    
    required_fields = [
        "name", "board_type", "squares_x", "squares_y",
        "square_length_m", "marker_length_m"
    ]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in board config")
    
    return BoardConfig(
        name=data["name"],
        board_type=data["board_type"],
        squares_x=int(data["squares_x"]),
        squares_y=int(data["squares_y"]),
        square_length_m=float(data["square_length_m"]),
        marker_length_m=float(data["marker_length_m"]),
        dictionary=data.get("dictionary", "DICT_6X6_250"),
        border_bits=data.get("border_bits", 1),
    )


def create_charuco_board(config: BoardConfig) -> Any:
    """Create a ChArUco board from configuration.
    
    Args:
        config: Board configuration.
        
    Returns:
        cv2.aruco.CharucoBoard instance.
        
    Raises:
        ImportError: If OpenCV is not available.
        ValueError: If the dictionary is not recognized.
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for ChArUco board creation")
    
    # Get ArUco dictionary
    dict_name = config.dictionary.upper()
    if not dict_name.startswith("DICT_"):
        dict_name = f"DICT_{dict_name}"
    
    if hasattr(aruco, dict_name):
        dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    else:
        raise ValueError(f"Unknown ArUco dictionary: {config.dictionary}")
    
    # Create board
    board = aruco.CharucoBoard(
        (config.squares_x, config.squares_y),
        config.square_length_m,
        config.marker_length_m,
        dictionary,
    )
    
    logger.debug(f"Created ChArUco board: {config.name}")
    return board


def detect_charuco_corners(
    image: NDArray[np.uint8],
    board: Any,
    camera_matrix: Optional[NDArray[np.float64]] = None,
    dist_coeffs: Optional[NDArray[np.float64]] = None,
) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.int32]]]:
    """Detect ChArUco corners in an image.
    
    Args:
        image: Grayscale or BGR image.
        board: cv2.aruco.CharucoBoard instance.
        camera_matrix: Optional camera matrix for corner refinement.
        dist_coeffs: Optional distortion coefficients.
        
    Returns:
        Tuple of (charuco_corners, charuco_ids) or (None, None) if not found.
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for corner detection")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create detector
    detector_params = aruco.DetectorParameters()
    dictionary = board.getDictionary()
    detector = aruco.ArucoDetector(dictionary, detector_params)
    
    # Detect ArUco markers
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    
    if marker_ids is None or len(marker_ids) == 0:
        logger.debug("No ArUco markers detected")
        return None, None
    
    # Interpolate ChArUco corners
    charuco_corners, charuco_ids, _, _ = aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )
    
    if charuco_ids is None or len(charuco_ids) < 4:
        logger.debug(f"Insufficient ChArUco corners detected: {len(charuco_ids) if charuco_ids is not None else 0}")
        return None, None
    
    logger.debug(f"Detected {len(charuco_ids)} ChArUco corners")
    return charuco_corners, charuco_ids


def draw_charuco_corners(
    image: NDArray[np.uint8],
    corners: NDArray[np.float64],
    ids: NDArray[np.int32],
) -> NDArray[np.uint8]:
    """Draw detected ChArUco corners on an image.
    
    Args:
        image: Input image (will be copied).
        corners: Detected corners.
        ids: Corner IDs.
        
    Returns:
        Image with drawn corners.
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for drawing")
    
    output = image.copy()
    aruco.drawDetectedCornersCharuco(output, corners, ids)
    return output
