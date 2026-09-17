"""Pattern visualization.

This module provides functions for visualizing the calibration pattern
and scan lines in pattern coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from hsi_rgbd_calib.viz.data_contracts import VisualizationData, ViewVizData


def get_pattern_bounds(
    wp1: float,
    wp2: float,
    margin: float = 0.1,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Get safe pattern drawing bounds.
    
    Uses fixed pattern dimensions with margin, not line intersections.
    
    Args:
        wp1: Pattern dimension 1 (meters).
        wp2: Pattern dimension 2 (meters).
        margin: Fractional margin (default 10%).
        
    Returns:
        ((x_min, x_max), (y_min, y_max)) in meters.
    """
    x_max = 2 * max(wp1, wp2)
    y_max = wp1
    return (
        (-margin * x_max, (1 + margin) * x_max),
        (-margin * y_max, (1 + margin) * y_max),
    )


def _line_endpoints_in_bounds(
    a: float, b: float, c: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Compute line endpoints within a bounding box.
    
    Line equation: ax + by + c = 0
    
    Args:
        a, b, c: Line coefficients.
        x_range: (x_min, x_max).
        y_range: (y_min, y_max).
        
    Returns:
        Two (x, y) points where line intersects bounds, or None.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    points = []
    eps = 1e-12
    
    # Intersect with x = x_min
    if abs(b) > eps:
        y = -(a * x_min + c) / b
        if y_min <= y <= y_max:
            points.append((x_min, y))
    
    # Intersect with x = x_max
    if abs(b) > eps:
        y = -(a * x_max + c) / b
        if y_min <= y <= y_max:
            points.append((x_max, y))
    
    # Intersect with y = y_min
    if abs(a) > eps:
        x = -(b * y_min + c) / a
        if x_min <= x <= x_max:
            points.append((x, y_min))
    
    # Intersect with y = y_max
    if abs(a) > eps:
        x = -(b * y_max + c) / a
        if x_min <= x <= x_max:
            points.append((x, y_max))
    
    # Remove duplicates with tolerance
    unique = []
    for p in points:
        is_dup = False
        for u in unique:
            if abs(p[0] - u[0]) < eps and abs(p[1] - u[1]) < eps:
                is_dup = True
                break
        if not is_dup:
            unique.append(p)
    
    if len(unique) >= 2:
        return (unique[0], unique[1])
    return None


def plot_pattern_with_scanlines(
    viz_data: VisualizationData,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 8),
    title: str = "Pattern with Scan Lines",
) -> plt.Figure:
    """Plot pattern feature lines with scan lines for each view.
    
    This is a core diagnostic plot showing:
    - The 6 feature lines of the Li-Wen-Qiu pattern
    - The scan line for each view (where HSI slit intersects pattern)
    - Recovered pattern points for each view
    
    Args:
        viz_data: Visualization data package.
        output_path: Optional path to save figure.
        figsize: Figure size.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get bounds
    x_range, y_range = get_pattern_bounds(viz_data.wp1, viz_data.wp2)
    
    # Draw pattern feature lines
    line_colors = plt.cm.Blues(np.linspace(0.4, 0.8, 6))
    for i, (a, b, c) in enumerate(viz_data.pattern_lines):
        endpoints = _line_endpoints_in_bounds(a, b, c, x_range, y_range)
        if endpoints:
            (x1, y1), (x2, y2) = endpoints
            ax.plot([x1, x2], [y1, y2], '-', color=line_colors[i], 
                    linewidth=2, label=f'L{i+1}' if i < 3 else None)
    
    # Draw scan lines and points for each view
    cmap = plt.cm.Set2
    for j, view in enumerate(viz_data.views):
        color = cmap(j % 8)
        a, b, c = view.scan_line
        
        # Draw scan line
        endpoints = _line_endpoints_in_bounds(a, b, c, x_range, y_range)
        if endpoints:
            (x1, y1), (x2, y2) = endpoints
            ax.plot([x1, x2], [y1, y2], '--', color=color, 
                    linewidth=1.5, alpha=0.7, label=f'{view.view_id}')
        
        # Draw recovered points
        P = view.P_pattern_final
        ax.scatter(P[:, 0], P[:, 1], c=[color], s=50, 
                   edgecolors='black', linewidth=0.5, zorder=10)
    
    # Formatting
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig
