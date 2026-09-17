"""Li-Wen-Qiu calibration method implementation.

This submodule contains the full implementation of the Li-Wen-Qiu
cross-ratio-based line-scan camera calibration method.

Reference:
    Li, Wen, Qiu (2016). Cross-ratio-based line scan camera calibration
    using a planar pattern. Optical Engineering 55(1), 014104.
    DOI: 10.1117/1.OE.55.1.014104
"""

from hsi_rgbd_calib.cal_method.li_wen_qiu.projection import (
    project_to_linescan,
    project_to_linescan_batch,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.cross_ratio import (
    compute_cross_ratios,
    compute_X3_X5_from_cross_ratios,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.closed_form import (
    closed_form_init,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.nonlinear import (
    refine_calibration,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.backend import (
    LiWenQiuBackend,
)

__all__ = [
    # Projection
    "project_to_linescan",
    "project_to_linescan_batch",
    # Cross-ratio
    "compute_cross_ratios",
    "compute_X3_X5_from_cross_ratios",
    # Closed-form
    "closed_form_init",
    # Nonlinear
    "refine_calibration",
    # Backend
    "LiWenQiuBackend",
]
