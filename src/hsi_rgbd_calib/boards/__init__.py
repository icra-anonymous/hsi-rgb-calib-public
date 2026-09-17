"""Calibration target board utilities."""

from hsi_rgbd_calib.boards.charuco import (
    BoardConfig,
    load_board_config,
    create_charuco_board,
    detect_charuco_corners,
)
from hsi_rgbd_calib.boards.target_metadata import (
    TargetDimensions,
    load_target_dimensions,
)
from hsi_rgbd_calib.boards.geometry import (
    normalize_line,
    line_through_points,
    intersect_lines_2d,
    cross_ratio_1d,
)
from hsi_rgbd_calib.boards.li_wen_qiu_pattern import (
    LiWenQiuPattern,
    load_li_wen_qiu_pattern,
    get_default_li_wen_qiu_pattern,
)

__all__ = [
    # ChArUco
    "BoardConfig",
    "load_board_config",
    "create_charuco_board",
    "detect_charuco_corners",
    # Target metadata
    "TargetDimensions",
    "load_target_dimensions",
    # Geometry
    "normalize_line",
    "line_through_points",
    "intersect_lines_2d",
    "cross_ratio_1d",
    # Li-Wen-Qiu pattern
    "LiWenQiuPattern",
    "load_li_wen_qiu_pattern",
    "get_default_li_wen_qiu_pattern",
]

