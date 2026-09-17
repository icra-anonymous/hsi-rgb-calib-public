"""3D rig visualization.

This module provides functions for visualizing the camera rig in 3D,
including camera frustums and pattern poses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hsi_rgbd_calib.viz.data_contracts import VisualizationData
from hsi_rgbd_calib.viz.transforms import CalibratedRig


def _draw_camera_frustum(
    ax: Axes3D,
    R: NDArray[np.float64],
    t: NDArray[np.float64],
    scale: float = 0.1,
    color: str = 'blue',
    label: Optional[str] = None,
    alpha: float = 0.3,
) -> None:
    """Draw a camera frustum in 3D.
    
    The frustum is drawn as a pyramid from the camera center.
    
    Args:
        ax: 3D axes to draw on.
        R: 3x3 rotation matrix (world→camera).
        t: 3-element camera position in world frame.
        scale: Size scale for the frustum.
        color: Color for the frustum.
        label: Optional label for legend.
        alpha: Transparency.
    """
    # Camera frame axes in world coordinates
    # R is world→camera, so R.T is camera→world
    x_axis = R.T @ np.array([1, 0, 0]) * scale
    y_axis = R.T @ np.array([0, 1, 0]) * scale
    z_axis = R.T @ np.array([0, 0, 1]) * scale
    
    # Frustum corners (in camera frame, then transformed)
    # Simple pyramid shape
    corners_cam = np.array([
        [0, 0, 0],  # apex (camera center)
        [-0.5, -0.5, 1],   # front corners
        [0.5, -0.5, 1],
        [0.5, 0.5, 1],
        [-0.5, 0.5, 1],
    ]) * scale
    
    corners_world = (R.T @ corners_cam.T).T + t
    
    # Draw frustum edges
    apex = corners_world[0]
    for i in range(1, 5):
        ax.plot3D([apex[0], corners_world[i][0]],
                  [apex[1], corners_world[i][1]],
                  [apex[2], corners_world[i][2]],
                  color=color, linewidth=1.5)
    
    # Connect front face
    for i in range(1, 5):
        j = i % 4 + 1
        ax.plot3D([corners_world[i][0], corners_world[j][0]],
                  [corners_world[i][1], corners_world[j][1]],
                  [corners_world[i][2], corners_world[j][2]],
                  color=color, linewidth=1.5)
    
    # Draw front face as polygon
    verts = [corners_world[1:5].tolist()]
    ax.add_collection3d(Poly3DCollection(verts, alpha=alpha, facecolor=color,
                                          edgecolor=color, linewidth=1))
    
    # Draw axes
    ax.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], 
              color='red', arrow_length_ratio=0.2)
    ax.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], 
              color='green', arrow_length_ratio=0.2)
    ax.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], 
              color='blue', arrow_length_ratio=0.2)
    
    if label:
        ax.text(t[0], t[1], t[2] + scale * 0.5, label, fontsize=8)


def _draw_pattern_quad(
    ax: Axes3D,
    R: NDArray[np.float64],
    t: NDArray[np.float64],
    wp1: float,
    wp2: float,
    color: str = 'gray',
    alpha: float = 0.2,
    label: Optional[str] = None,
) -> None:
    """Draw the pattern as a quadrilateral in 3D.
    
    Args:
        ax: 3D axes.
        R: 3x3 rotation pattern→world.
        t: 3-element pattern origin in world.
        wp1, wp2: Pattern dimensions.
        color: Fill color.
        alpha: Transparency.
        label: Optional label.
    """
    # Pattern corners in pattern frame (Z=0)
    x_max = 2 * max(wp1, wp2)
    y_max = wp1
    corners_pattern = np.array([
        [0, 0, 0],
        [x_max, 0, 0],
        [x_max, y_max, 0],
        [0, y_max, 0],
    ])
    
    # Transform to world
    corners_world = (R @ corners_pattern.T).T + t
    
    verts = [corners_world.tolist()]
    ax.add_collection3d(Poly3DCollection(verts, alpha=alpha, facecolor=color,
                                          edgecolor='black', linewidth=1))
    
    if label:
        center = corners_world.mean(axis=0)
        ax.text(center[0], center[1], center[2], label, fontsize=8)


def plot_rig_3d(
    viz_data: VisualizationData,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 8),
    title: str = "3D Camera Rig Visualization",
    max_views: int = 10,
) -> plt.Figure:
    """Plot 3D visualization of camera rig with pattern poses.
    
    Shows:
    - OAK-RGB camera at origin (world frame)
    - HSI camera offset by extrinsics
    - Pattern poses for each view
    
    Args:
        viz_data: Visualization data package.
        output_path: Optional path to save figure.
        figsize: Figure size.
        title: Plot title.
        max_views: Maximum number of view patterns to show.
        
    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create CalibratedRig
    rig = CalibratedRig(viz_data.T_oakrgb_hsi)
    
    # Draw OAK-RGB camera at origin (identity)
    _draw_camera_frustum(
        ax, np.eye(3), np.zeros(3), 
        scale=0.05, color='blue', label='OAK-RGB'
    )
    
    # Draw HSI camera
    # HSI origin in OAK-RGB frame
    hsi_pos = rig.hsi_origin_in_oakrgb()
    hsi_R = viz_data.T_oakrgb_hsi[:3, :3]  # already world→HSI rotation
    _draw_camera_frustum(
        ax, hsi_R.T, hsi_pos,  # hsi_R.T converts HSI→OAK-RGB back to OAK-RGB→HSI
        scale=0.05, color='orange', label='HSI'
    )
    
    # Draw pattern poses for each view
    cmap = plt.cm.Set2
    n_views = min(len(viz_data.views), max_views)
    
    for j in range(n_views):
        view = viz_data.views[j]
        color = cmap(j % 8)
        
        # Pattern pose in frame-camera frame -> we want in OAK-RGB (world)
        # R_frame_pattern, T_frame_pattern are pattern→frame-camera
        # Pattern origin in OAK-RGB: just T_frame_pattern (since frame-cam ≈ OAK-RGB)
        R = view.R_frame_pattern
        t = view.T_frame_pattern
        
        _draw_pattern_quad(
            ax, R, t, viz_data.wp1, viz_data.wp2,
            color=color, alpha=0.15, label=view.view_id if j < 5 else None
        )
    
    # Set equal aspect ratio
    all_pts = [np.zeros(3), hsi_pos]
    for v in viz_data.views[:n_views]:
        all_pts.append(v.T_frame_pattern)
    all_pts = np.array(all_pts)
    
    max_range = np.ptp(all_pts, axis=0).max() / 2.0 + 0.1
    mid = all_pts.mean(axis=0)
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"{title}\n(World = OAK-RGB frame)")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig
