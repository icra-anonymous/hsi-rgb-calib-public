#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import copy

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsi_rgbd_calib.cal_method.li_wen_qiu import LiWenQiuBackend
from hsi_rgbd_calib.cal_method.li_wen_qiu.backend import ViewObservation
import yaml

def main():
    ap = argparse.ArgumentParser(description="Phase 2: Estimate HSI-RGB extrinsics from detections")
    ap.add_argument("--detections", type=Path, default=Path("output/phase1/detections.json"))
    ap.add_argument("--target-config", type=Path, default=Path("assets/calibration_targets/combined_target.yaml"))
    ap.add_argument("--calib-json", type=Path, required=True, help="OAK-D calibration JSON")
    ap.add_argument("--out", type=Path, default=Path("output/phase2"))
    ap.add_argument("--physical-init", action="store_true",
                    help="Bypass closed-form init and use a physically-estimated R,T as starting point")
    ap.add_argument("--hsi-rz-init-deg", type=float, default=90.0,
                    help="Initial guess for the rotation (deg, about Z) between RGB and HSI frames. "
                         "Used with --physical-init. 0 = identity (HSI-Y // RGB-Y, no twist between "
                         "the two camera mounts). 90 = the previous hardcoded default (HSI-Y // RGB-X). "
                         "-90 is also valid if the rig is twisted the other way. Default: 90")
    ap.add_argument("--hsi-above-cm", type=float, default=6.0,
                    help="Physical offset of HSI above RGB camera (cm). Used with --physical-init. Default: 6.0")
    ap.add_argument("--f-init", type=float, default=800.0,
                    help="Initial focal length guess for --physical-init. Default: 800")
    ap.add_argument("--v0-init", type=float, default=640.0,
                    help="Initial principal point guess for --physical-init. Default: 640")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Load target and initialize backend
    #    Target feature lines and intersections for LiWenQiu
    print(f"Loading target config: {args.target_config}")
    import yaml
    from hsi_rgbd_calib.boards.li_wen_qiu_pattern import LiWenQiuPattern
    with open(args.target_config, 'r') as f:
        target_cfg_raw = yaml.safe_load(f)
    
    lwq_data = target_cfg_raw.get("li_wen_qiu", target_cfg_raw)
    
    # combined_target.yaml has a different schema than the default parser expects
    feature_lines = []
    line_names = []
    for i in range(1, 7):
        name = f"L{i}"
        line_names.append(name)
        
        # The line equations in YAML [a, b, c] are for coordinates in mm: a*x + b*y + c = 0.
        # Rewrite in meters (divide all 3 by 1000 to keep the line homogeneous):
        # (a/1000)*x_m + (b/1000)*y_m + (c/1000000) = 0  (equivalent but consistent scaling)
        eq_mm = lwq_data["feature_lines"][name]["eq"]
        a, b, c = eq_mm[0], eq_mm[1], eq_mm[2]
        
        feature_lines.append( (float(a)/1000.0, float(b)/1000.0, float(c)/1_000_000.0) )
    
    target_cfg = LiWenQiuPattern(
        name="custom_target",
        wp1=lwq_data.get("wp_mm", 240.0) / 1000.0,
        wp2=lwq_data.get("hp_mm", 60.0) / 1000.0,
        feature_lines=feature_lines,
        line_names=line_names
    )
    backend = LiWenQiuBackend(target_cfg)
    
    # 2. Load Phase 1 detections
    if not args.detections.exists():
        print(f"Error: detections file not found at {args.detections}")
        return
        
    print(f"Loading Phase 1 detections from: {args.detections}")
    with open(args.detections, "r") as f:
        detections = json.load(f)

    # The HSI spatial axis (which raw-image axis the pushbroom scan runs
    # along) is a property of the whole capture session, not per-view, but
    # it's stored per-detection -- check it's consistent and carry it through
    # to the saved extrinsics, since downstream HSI mapping needs to know it.
    # A missing field (older detections.json) defaults for backward
    # compatibility, but that default may not match how Phase 1 was actually
    # run, so it's flagged loudly rather than silently written as fact.
    if any("spatial_axis" not in det for det in detections):
        print("  WARNING: some detections in detections.json have no 'spatial_axis' "
              "(re-run detect_and_visualize.py to record it); assuming 'width'. "
              "If Phase 1 was actually run with --spatial-axis height, this will be WRONG.")
    spatial_axes = {det.get("spatial_axis", "width") for det in detections}
    if len(spatial_axes) > 1:
        print(f"  WARNING: detections.json has mixed spatial_axis values {sorted(spatial_axes)}; "
              f"using '{sorted(spatial_axes)[0]}' for the saved extrinsics file.")
    dataset_spatial_axis = sorted(spatial_axes)[0] if spatial_axes else "width"

    # Same idea for hsi_scale: the --hsi-scale Phase 1 was run with, which
    # f_px/v0_px below are expressed in. Needed alongside spatial_axis to
    # index the *raw* (unscaled) HSI image correctly during HSI mapping.
    if any("hsi_scale" not in det for det in detections):
        print("  WARNING: some detections in detections.json have no 'hsi_scale' "
              "(re-run detect_and_visualize.py to record it); assuming 1.0 (no "
              "scaling). If Phase 1 was actually run with --hsi-scale != 1.0, "
              "this will be WRONG.")
    hsi_scales = {det.get("hsi_scale", 1.0) for det in detections}
    if len(hsi_scales) > 1:
        print(f"  WARNING: detections.json has mixed hsi_scale values {sorted(hsi_scales)}; "
              f"using {sorted(hsi_scales)[0]} for the saved extrinsics file.")
    dataset_hsi_scale = sorted(hsi_scales)[0] if hsi_scales else 1.0

    # 3. Load intrinsic parameters
    # The detect_and_visualize scaled intrinsics for 720p already, 
    # but run_calibration only uses them for visualization. 
    # The actual 3D points in the RGB frame are given by R, T inside detections!
    with open(args.calib_json, 'r') as f:
        calib_data = json.load(f)
    if "rgb" in calib_data and "intrinsic_matrix_3x3" in calib_data["rgb"]:
        K = np.array(calib_data["rgb"]["intrinsic_matrix_3x3"])
    elif "cameraData" in calib_data and "rgb" in calib_data["cameraData"]:
        rgb_data = calib_data["cameraData"]["rgb"]
        K = np.array(rgb_data["intrinsicMatrix"])
    elif "cameraData" in calib_data and "0" in calib_data["cameraData"]:
        rgb_data = calib_data["cameraData"]["0"] # Sometimes OAK-D uses '0' for RGB CAM_A
        K = np.array(rgb_data["intrinsicMatrix"])
    elif "board_config" in calib_data and "cameras" in calib_data["board_config"]:
        K = np.array(calib_data["board_config"]["cameras"]["rgb"]["intrinsics"])
    else:
        print("Error: Could not find RGB intrinsics in calibration.json")
        return
    
    # K needs to be scaled to 1280x720 for overlaid visualization
    K_scaled = K.copy()
    K_scaled[0, 0] *= (1280 / 1920.0)
    K_scaled[1, 1] *= (720 / 1080.0)
    K_scaled[0, 2] *= (1280 / 1920.0)
    K_scaled[1, 2] *= (720 / 1080.0)

    # 4. Create observations
    observations = []
    print("\nProcessing Views...")
    for det in detections:
        view_id = det["view_id"]
        if not det.get("pose_found", False):
            print(f"  Skipping {view_id}: No RGB pose found")
            continue
            
        # The observed edges u in the HSI image (we call them v_c in our lit)
        v_obs = det["v_observations"]
        if len(v_obs) != 6:
            print(f"  Skipping {view_id}: Expected 6 HSI edges, found {len(v_obs)}")
            continue

        # RGB pose of the calibration target (board to camera)
        R_c2b = np.array(det["R"])
        t_c2b = np.array(det["T"]).reshape((3, 1))

        # Ensure v_obs is exactly 6 points
        if len(v_obs) != 6:
            continue
            
        # The interactive tool in Phase 1 sorted the points purely by X-coordinate (left to right).
        # We need to map these 6 points to the L1..L6 feature lines of the pattern.
        # In the Li-Wen-Qiu pattern:
        # L1, L2, L3 are horizontal lines (Y=0, Y=wp2, Y=wp1)
        # L4, L5, L6 are diagonal lines (X=Y, X-Y=wp2, X-Y=wp1)
        # 
        # The updated interactive tool explicitly enforces the user to map L1..L6
        # to the correct visual lines, so the exported v_obs array is already
        # strictly in the required order [L1, L2, L3, L4, L5, L6].
        
        v_reordered = np.array(v_obs, dtype=np.float64)
        
        print(f"  View {view_id}: loaded L1..L6 direct observations.")
        
        obs = ViewObservation(
            R_frame_pattern=R_c2b,
            T_frame_pattern=t_c2b.ravel(),
            v_observations=v_reordered,
            view_id=view_id
        )
        observations.append(obs)
        print(f"  Added {view_id}")

    if len(observations) < 3:
        print(f"\nError: Need ≥ 3 valid views. Only got {len(observations)}")
        return

    # 5. Run calibration
    print("\n--- Running Extrinsics Estimation (Li-Wen-Qiu) ---")

    if args.physical_init:
        # ---------------------------------------------------------------
        # Physical-prior initialization path
        # Bypasses the (often-failing) closed-form cross-ratio step and
        # directly seeds the nonlinear optimizer with our engineering estimate:
        #
        #  R_init: rotation around Z between the RGB and HSI mounts, by
        #          --hsi-rz-init-deg (default 90 -> HSI-Y // RGB-X):
        #            | cos  -sin  0 |
        #            | sin   cos  0 |
        #            |  0     0   1 |
        #          0 deg gives identity (HSI-Y // RGB-Y, no twist).
        #
        #  T_init: HSI is ~6 cm above RGB  ->  [0, -0.06, 0]  (Y points down)
        # ---------------------------------------------------------------
        from hsi_rgbd_calib.cal_method.li_wen_qiu.nonlinear import refine_calibration

        theta = np.radians(args.hsi_rz_init_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_phys = np.array([[cos_t, -sin_t, 0.],
                           [sin_t,  cos_t, 0.],
                           [0.,     0.,    1.]], dtype=np.float64)
        # T_rgb2hsi: origin of RGB in HSI frame = -R @ [0, -hsi_above, 0]
        T_phys = -(R_phys @ np.array([0., -args.hsi_above_cm / 100.0, 0.]))

        frame_poses = [(obs.R_frame_pattern, obs.T_frame_pattern) for obs in observations]
        v_obs_list  = [obs.v_observations for obs in observations]

        print(f"  Physical R_init ({args.hsi_rz_init_deg:g}deg around Z):\n{R_phys}")
        print(f"  Physical T_init (HSI {args.hsi_above_cm} cm above RGB): {T_phys}")
        print(f"  f_init={args.f_init},  v0_init={args.v0_init}")

        refine_result = refine_calibration(
            R_init       = R_phys,
            T_init       = T_phys,
            f_init       = args.f_init,
            v0_init      = args.v0_init,
            pattern_lines= target_cfg.feature_lines,
            frame_poses  = frame_poses,
            v_observations= v_obs_list,
            k_init       = 0.0,
            max_iter     = 5000,
            tol          = 1e-6,
        )

        print(f"\n  Optimizer: {refine_result.message}")
        print(f"  Initial cost: {refine_result.initial_cost:.4f}  ->  Final cost: {refine_result.final_cost:.4f}")
        print(f"  Cost reduction: {(refine_result.initial_cost - refine_result.final_cost) / refine_result.initial_cost * 100:.2f}%")

        R_rgb2hsi = refine_result.R
        t_rgb2hsi = refine_result.T.reshape(3, 1)
        f_est  = refine_result.f
        v0_est = refine_result.v0
        k_est  = refine_result.k

        # Invert to get HSI -> RGB which is physically easier to read
        R_h2c = R_rgb2hsi.T
        t_h2c = -R_rgb2hsi.T @ t_rgb2hsi

        print("\nResulting Extrinsics (HSI -> RGB):")
        print("R (Rotation):\n", R_h2c)
        print("T (Translation, meters):\n", t_h2c)
        print(f"HSI intrinsics: f={f_est:.2f} px,  v0={v0_est:.2f} px,  k={k_est:.6f}")

    else:
        # ---------------------------------------------------------------
        # Default: full backend pipeline (closed-form + nonlinear)
        # ---------------------------------------------------------------
        from hsi_rgbd_calib.cal_method.interface import CalibrationConfig
        config = CalibrationConfig.from_dict({
            "max_iterations": 5000,
            "convergence_threshold": 1e-6,
        })

        ret = backend.estimate_from_observations(observations, config)
        if not ret.success:
            print(f"Calibration failed: optimization did not converge.")
            if hasattr(ret, "message"):
                print(f"Reason: {ret.message}")
            return

        print(f"\nCalibration Succeeded!")
        T_rgb_hsi = ret.T_oakrgb_hsi
        R_h2c = T_rgb_hsi[:3, :3]
        t_h2c = T_rgb_hsi[:3, 3:]
        f_est  = ret.hsi_intrinsics.focal_length_slit
        v0_est = ret.hsi_intrinsics.principal_point_u0
        k_est  = getattr(ret.hsi_intrinsics, 'k', 0.0)  # fallback if not present

        print("\nResulting Extrinsics (HSI -> RGB):")
        print("R (Rotation):\n", R_h2c)
        print("T (Translation, meters):\n", t_h2c)

    # 6. Save results
    calib_res = {
        "R_h2c": R_h2c.tolist(),
        "t_h2c": t_h2c.ravel().tolist(),
        "hsi_intrinsics": {
            "f_px": float(f_est),
            "v0_px": float(v0_est),
            "k": float(k_est),
            "note": "f_px is the HSI line-scan focal length (can be negative if pixel readout axis is inverted). v0_px is the principal point. k is the radial distortion along the 1D sensor."
        },
        "spatial_axis": dataset_spatial_axis,
        "spatial_axis_note": (
            "Which raw HSI/FLIR image axis the pushbroom spatial dimension (and "
            "therefore v0_px / f_px / the v pixel coordinate) runs along: "
            "'width' = columns, 'height' = rows. Needed to index the raw image "
            "correctly when mapping HSI pixels."
        ),
        "hsi_scale": dataset_hsi_scale,
        "hsi_scale_note": (
            "The --hsi-scale Phase 1 (detect_and_visualize.py) was run with. "
            "f_px, v0_px, and any v pixel coordinate above are all expressed in "
            "this scaled resolution -- multiply by 1/hsi_scale to map them onto "
            "the raw, full-resolution HSI image."
        ),
    }
    calib_out = args.out / "hsi_rgb_extrinsics.json"
    with open(calib_out, "w") as f:
        json.dump(calib_res, f, indent=2)
    print(f"\nSaved extrinsics to {calib_out}")

    # 7. Visualization: Project HSI slit into each RGB image
    print("\n--- Generating Reprojection Visualizations ---")
    for det in detections:
        if not det.get("pose_found", False):
            continue
            
        rgb_path = det["rgb_file"]
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            continue
            
        view_id = det["view_id"]
        ts = det["timestamp"]
        v_obs = det["v_observations"]
        
        R_board = np.array(det["R"])
        T_board = np.array(det["T"]).reshape((3, 1))

        # The HSI camera (line-scan) only sees a single 3D plane in space: X_h = 0.
        # Let's define this slit plane in the RGB frame: n_c^T * P_rgb + d_c = 0
        n_h = np.array([[1.0], [0.0], [0.0]])
        n_c = R_h2c @ n_h
        d_c = -(n_c.T @ t_h2c)[0, 0]

        # The calibration board defines another 3D plane in the RGB frame:
        # P_rgb = R_board * P_board + T_board, where P_board = [X_b, Y_b, 0]^T
        #
        # By intersecting these two planes, we find the exact 3D line on the board 
        # that the HSI slit is currently laser-scanning.
        #
        # n_c^T * (R_board * P_board + T_board) + d_c = 0
        # (n_c^T * R_board[:, 0]) * X_b + (n_c^T * R_board[:, 1]) * Y_b + (n_c^T * T_board + d_c) = 0
        A = (n_c.T @ R_board[:, 0:1])[0, 0]
        B = (n_c.T @ R_board[:, 1:2])[0, 0]
        C = (n_c.T @ T_board)[0, 0] + d_c

        # Find two 3D points on this intersection line in board coordinates
        pts_3d_rgb = []
        if abs(B) > 1e-4:
            # Sample at the left and right edges of the ArUco markers
            # Markers 0 & 2 left edge: -23.5mm, Markers 1 & 3 right edge: 263.5mm
            for x_b in [-0.0235, 0.2635]:
                y_b = -(A * x_b + C) / B
                P_b = np.array([[x_b], [y_b], [0.0]])
                P_c = R_board @ P_b + T_board
                pts_3d_rgb.append(P_c)
        else:
            # Vertical line in the board frame
            x_b = -C / A
            for y_b in [0.0, 0.18]:
                P_b = np.array([[x_b], [y_b], [0.0]])
                P_c = R_board @ P_b + T_board
                pts_3d_rgb.append(P_c)

        # Project these two 3D intersection points into the 2D RGB image
        pts_2d = []
        for P_c in pts_3d_rgb:
            p_img = K_scaled @ P_c
            u_px = int(round(p_img[0, 0] / p_img[2, 0]))
            v_px = int(round(p_img[1, 0] / p_img[2, 0]))
            pts_2d.append((u_px, v_px))

        # ── Draw Naive Projection (Baseline) ────────────────────────────────────
        # Naive model: assume HSI and RGB camera origins are identical (T=0)
        # and only a pure 90-degree around Z rotation exists.
        # R_naive = [[0,-1,0],[1,0,0],[0,0,1]]
        # Slit normal in RGB frame = R_naive @ [1,0,0] = [0, 1, 0]
        n_naive = np.array([[0.0], [1.0], [0.0]])
        d_naive = 0.0

        A_n = (n_naive.T @ R_board[:, 0:1])[0, 0]
        B_n = (n_naive.T @ R_board[:, 1:2])[0, 0]
        C_n = (n_naive.T @ T_board)[0, 0] + d_naive

        pts_3d_naive = []
        if abs(B_n) > 1e-4:
            for x_b in [-0.0235, 0.2635]:
                y_b = -(A_n * x_b + C_n) / B_n
                P_b = np.array([[x_b], [y_b], [0.0]])
                pts_3d_naive.append(R_board @ P_b + T_board)
        else:
            x_b = -C_n / A_n
            for y_b in [0.0, 0.18]:
                P_b = np.array([[x_b], [y_b], [0.0]])
                pts_3d_naive.append(R_board @ P_b + T_board)

        pts_2d_naive = []
        for P_c in pts_3d_naive:
            p_img = K_scaled @ P_c
            pts_2d_naive.append( (int(round(p_img[0, 0] / p_img[2, 0])),
                                  int(round(p_img[1, 0] / p_img[2, 0]))) )

        # Draw naive as cyan dashed line
        pass  # Will draw after cropping to keep coordinate systems manageable

        # Find 2D bounds of the board
        corners_3d = []
        for xb, yb in [(0,0), (0.24,0), (0.24,0.18), (0,0.18)]:
            corners_3d.append(R_board @ np.array([[xb],[yb],[0]]) + T_board)
        
        corners_2d = []
        for P_c in corners_3d:
            p = K_scaled @ P_c
            corners_2d.append([int(p[0,0]/p[2,0]), int(p[1,0]/p[2,0])])
        corners_2d = np.array(corners_2d)
        
        # Determine strict 2D x-bounds for the marker area
        x_min_2d = np.min(corners_2d[:, 0])
        x_max_2d = np.max(corners_2d[:, 0])

        def _clip_line_x2d(pt1, pt2, x_min, x_max):
            """Clips a 2D line defined by pt1 and pt2 to the range [x_min, x_max]."""
            x1, y1 = pt1
            x2, y2 = pt2
            if x1 == x2:
                return pt1, pt2
            
            # Sort by x
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            # Line equation
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            
            # Clip
            nx1 = max(x1, x_min)
            ny1 = m * nx1 + b
            nx2 = min(x2, x_max)
            ny2 = m * nx2 + b
            
            return (int(nx1), int(ny1)), (int(nx2), int(ny2))

        # Draw calibrated line
        if len(pts_2d) == 2:
            p1, p2 = _clip_line_x2d(pts_2d[0], pts_2d[1], x_min_2d, x_max_2d)
            cv2.line(rgb_img, p1, p2, (0, 255, 255), 3) # Calibrated
            
        # Draw dashed cyan line for naive
        if len(pts_2d_naive) == 2:
            p1, p2 = _clip_line_x2d(pts_2d_naive[0], pts_2d_naive[1], x_min_2d, x_max_2d)
            import math
            x1, y1 = p1
            x2, y2 = p2
            dist = math.hypot(x2-x1, y2-y1)
            dash_length = 8
            if dist > 0:
                for i in range(0, int(dist), dash_length*2):
                    alpha = i/dist
                    beta = min((i+dash_length)/dist, 1.0)
                    px1 = int(x1 + alpha*(x2-x1))
                    py1 = int(y1 + alpha*(y2-y1))
                    px2 = int(x1 + beta*(x2-x1))
                    py2 = int(y1 + beta*(y2-y1))
                    cv2.line(rgb_img, (px1, py1), (px2, py2), (255, 255, 0), 2) # Cyan

        dist_zero = np.zeros(5)
        cv2.drawFrameAxes(rgb_img, K_scaled, dist_zero, R_board, T_board, 0.05)

        # Remove text overlays for paper (will be in caption) and keep uncropped
        out_img_path = args.out / f"{view_id}_slit_projection.png"
        cv2.imwrite(str(out_img_path), rgb_img)
        print(f"  Saved projection -> {out_img_path}")

    # ── 8b. Visualize reprojection error in HSI images ──────────────────────
    print("\n--- Generating HSI Reprojection Error Visualizations ---")

    # Re-compute R_rgb2hsi from the saved R_h2c  (R_rgb2hsi = R_h2c.T)
    R_rgb2hsi = R_h2c.T
    T_rgb2hsi = (-R_h2c.T @ t_h2c).ravel()

    from hsi_rgbd_calib.cal_method.li_wen_qiu.projection import (
        compute_transform_pattern_to_linescan,
        compute_scan_line_in_pattern,
        project_to_linescan,
    )
    from hsi_rgbd_calib.boards.geometry import intersect_lines_2d

    pattern_lines = target_cfg.feature_lines  # list of (a,b,c) in meters
    warned_missing_band = False

    for det in detections:
        if not det.get("pose_found"):
            continue
        view_id = det["view_id"]
        v_obs   = np.array(det["v_observations"], dtype=np.float64)
        R_board = np.array(det["R"], dtype=np.float64)
        T_board = np.array(det["T"], dtype=np.float64).reshape(3, 1)

        # Load HSI image (stored as flir_file in detections)
        hsi_path = det.get("flir_file", "")
        hsi_img = cv2.imread(hsi_path) if hsi_path else None
        if hsi_img is None:
            print(f"  Skipping {view_id}: HSI image not found at {hsi_path}")
            continue

        # v_obs/v_pred and all the drawing math below were written assuming the
        # pushbroom spatial axis runs along image *columns* ("width" mode).
        # For "height" mode (spatial = rows), transpose into that canonical
        # orientation, do everything unchanged, then transpose the finished
        # crop back to the image's true orientation right before saving.
        spatial_axis = det.get("spatial_axis", "width")
        is_width = spatial_axis == "width"
        if not is_width:
            hsi_img = cv2.transpose(hsi_img)

        h_hsi, w_hsi = hsi_img.shape[:2]

        # Compute predicted v values for each feature line
        R0, T0 = compute_transform_pattern_to_linescan(R_board, T_board.ravel(),
                                                        R_rgb2hsi, T_rgb2hsi)
        scan_line = compute_scan_line_in_pattern(R0, T0)

        line_colors = [
            (0,   255,   0),   # L1 green
            (0,   200, 255),   # L2 cyan
            (255, 180,   0),   # L3 orange
            (200,   0, 255),   # L4 purple
            (0,   255, 180),   # L5 teal
            (255,  80,  80),   # L6 red-pink
        ]

        # ── 1D Intensity Profile + Reprojection (Sub-pixel obvious) ─────────
        # Average the FLIR image vertically to get the 1D intensity profile used in detection
        if len(hsi_img.shape) == 3:
            hsi_gray = cv2.cvtColor(hsi_img, cv2.COLOR_BGR2GRAY)
        else:
            hsi_gray = hsi_img
            
        SCALE = 2.0  # From Phase 1 50% scale

        # Band (perpendicular to the spatial axis) that Phase 1 actually
        # averaged over, in this canonical orientation -- i.e. a row range
        # here, whether that was rows (width mode) or columns (height mode,
        # before run_calibration's own transpose above) in the raw file.
        # Falls back to the old hardcoded guess for detections.json files
        # saved before this field existed.
        band = det.get("band_range")
        if band is None:
            band = (400, 600)
            if not warned_missing_band:
                print("  WARNING: detections.json has no 'band_range' (re-run "
                      "detect_and_visualize.py to get one); falling back to a "
                      "guessed (400, 600) band for the profile backdrop.")
                warned_missing_band = True
        band_lo = max(0, int(band[0] * SCALE))
        band_hi = min(h_hsi, int(band[1] * SCALE))
        profile_1d = np.mean(hsi_gray[band_lo:band_hi, :], axis=0)

        # Increase contrast/brightness for the HSI image to make it clearer for the paper
        hsi_img = cv2.convertScaleAbs(hsi_img, alpha=1.8, beta=30)

        # Row where the L1..L6 labels/stems sit.
        label_y = 350

        # Max height for plotting the profile, placed near the true bottom of
        # the image (not a fixed absolute row -- that clipped or floated
        # depending on the actual image size).
        plot_h = min(300, max(50, h_hsi // 4))
        y_offset = h_hsi - plot_h - 20

        # Normalize profile for plotting
        p_min, p_max = profile_1d.min(), profile_1d.max()
        if p_max > p_min:
            scaled_profile = (profile_1d - p_min) / (p_max - p_min) * (plot_h - 10)
        else:
            scaled_profile = np.zeros_like(profile_1d)

        # Draw the 1D profile
        pts = []
        for x, y in enumerate(scaled_profile):
            pts.append([x, y_offset + plot_h - 5 - int(y)])
        pts = np.array(pts, np.int32)

        # Draw solid white background for the plot area
        cv2.rectangle(hsi_img, (0, y_offset), (w_hsi, y_offset + plot_h), (255, 255, 255), -1)

        # Draw dark gray profile line
        cv2.polylines(hsi_img, [pts], False, (50, 50, 50), 2)

        # In height mode, text baked in now would come out sideways/mirrored
        # once we transpose back to the true orientation below. Defer these
        # labels and draw them upright afterward, directly onto the final crop.
        pending_labels = []

        # First pass: compute every line's observed/predicted position before
        # drawing, so the crop below can be sized around wherever the markers
        # actually land instead of a fixed window tuned to one dataset.
        line_results = []
        for i, (feat_line, color) in enumerate(zip(pattern_lines, line_colors)):
            a, b, c = feat_line
            feat_line_np = np.array([a, b, c])
            scan_np      = np.array(scan_line)
            pt = intersect_lines_2d(scan_np, feat_line_np)
            v_pred = np.nan
            if pt is not None:
                P_board = np.array([pt[0], pt[1], 0.0])
                v_pred = project_to_linescan(P_board, R0, T0, f_est, v0_est, k_est)

            v_o_col = v_obs[i] * SCALE
            v_p_col = v_pred * SCALE if not np.isnan(v_pred) else None
            line_results.append((i, color, v_o_col, v_p_col))

        # Adaptive column crop: cover every observed/predicted marker (with
        # margin) instead of a fixed window that only happened to fit one
        # historical dataset's marker range.
        all_cols = [v_o for _, _, v_o, _ in line_results]
        all_cols += [v_p for _, _, _, v_p in line_results if v_p is not None]
        margin = 150
        if all_cols:
            c_min = max(0, int(min(all_cols)) - margin)
            c_max = min(w_hsi, int(max(all_cols)) + margin)
        else:
            c_min, c_max = 0, w_hsi
        r_min = max(0, label_y - 30)
        r_max = min(h_hsi, y_offset + plot_h + 10)

        # Second pass: actually draw, now that layout is decided.
        for i, color, v_o_col, v_p_col in line_results:
            # Numbering for the line (L1 to L6 right to left in the image, meaning i=0 is L1)
            line_idx = i + 1

            # Draw Measured (Observed) marker on the profile & solid line
            if 0 <= v_o_col < w_hsi:
                profile_val = scaled_profile[int(min(max(v_o_col, 0), w_hsi-1))]
                pt_y = y_offset + plot_h - 5 - int(profile_val)
                cv2.circle(hsi_img, (int(v_o_col), pt_y), 6, color, -1)

                # Solid line going up
                cv2.line(hsi_img, (int(v_o_col), label_y + 20), (int(v_o_col), pt_y), color, 2)

                # Label L1-L6
                label_x = int(v_o_col) - 15
                if is_width:
                    cv2.putText(hsi_img, f"L{line_idx}", (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
                else:
                    pending_labels.append((f"L{line_idx}", label_x, label_y, color))

            # Draw Predicted marker on profile & dashed line
            if v_p_col is not None and 0 <= v_p_col < w_hsi:
                profile_val = scaled_profile[int(min(max(v_p_col, 0), w_hsi-1))]
                pt_y = y_offset + plot_h - 5 - int(profile_val)

                # Hollow diamond
                cv2.drawMarker(hsi_img, (int(v_p_col), pt_y), color,
                               markerType=cv2.MARKER_DIAMOND, markerSize=14, thickness=2)

                # Dashed line going up
                for gy0 in range(label_y + 20, pt_y, 40):
                    gy1 = min(gy0 + 20, pt_y)
                    cv2.line(hsi_img, (int(v_p_col), gy0), (int(v_p_col), gy1), color, 2)
        
        hsi_crop = hsi_img[r_min:r_max, c_min:c_max]

        # Undo the canonical-orientation transpose so the saved image matches
        # the true orientation of the raw HSI capture.
        if not is_width:
            hsi_crop = cv2.transpose(hsi_crop)
            # Draw the deferred labels now, upright, in the final orientation.
            # Transpose maps canonical (x, y) -> final (y, x); the crop offset
            # (c_min, r_min) must be subtracted before that swap.
            for text, tx, ty, color in pending_labels:
                fx, fy = ty - r_min, tx - c_min
                cv2.putText(hsi_crop, text, (fx, fy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        # No title text on the image itself, paper captions will cover it

        out_hsi_path = args.out / f"{view_id}_hsi_reproj.png"
        cv2.imwrite(str(out_hsi_path), hsi_crop)
        print(f"  Saved HSI reproj -> {out_hsi_path}")

    # ── 8c. 3D Rig Visualization ─────────────────────────────────────────────
    print("\n--- Generating 3D Rig Visualization ---")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def _frustum(ax, R_cam2world, t_world, scale=0.05, color='blue', label=None):
        """Draw a camera pyramid + XYZ axes."""
        # Frustum corners in camera frame
        cc = np.array([[0,0,0],[-0.5,-0.5,1],[0.5,-0.5,1],
                       [0.5,0.5,1],[-0.5,0.5,1]]) * scale
        cw = (R_cam2world @ cc.T).T + t_world
        apex = cw[0]
        for i in range(1, 5):
            ax.plot3D(*zip(apex, cw[i]), color=color, lw=1.5)
        for i in range(1, 5):
            ax.plot3D(*zip(cw[i], cw[i % 4 + 1]), color=color, lw=1.5)
        face = [cw[1:5].tolist()]
        ax.add_collection3d(Poly3DCollection(face, alpha=0.25,
                                             facecolor=color, edgecolor=color))
        for axis, clr in zip(np.eye(3), ['red','green','blue']):
            d = R_cam2world @ axis * scale
            ax.quiver(*t_world, *d, color=clr, arrow_length_ratio=0.3)
        if label:
            ax.text(*(t_world + np.array([0, 0, scale * 0.6])), label, fontsize=9,
                    fontweight='bold')

    def _board(ax, R, T, size_x=0.24, size_y=0.12, color='gray', label=None):
        """Draw the calibration board as a filled quad."""
        T = np.array(T).ravel()
        corners_b = np.array([[0,0,0],[size_x,0,0],[size_x,size_y,0],[0,size_y,0]])
        corners_w = (R @ corners_b.T).T + T
        face = [corners_w.tolist()]
        ax.add_collection3d(Poly3DCollection(face, alpha=0.25,
                                             facecolor=color, edgecolor='k', lw=0.8))
        # Draw board X and Y axes
        for axis, clr in zip(np.eye(3)[:2], ['crimson','limegreen']):
            d = R @ axis * 0.04
            ax.quiver(*T, *d, color=clr, arrow_length_ratio=0.4, lw=1)
        if label:
            ctr = corners_w.mean(axis=0)
            ax.text(*ctr, label, fontsize=7, ha='center')

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection='3d')

    # ── RGB camera at world origin ────────────────────────────────────────────
    _frustum(ax, np.eye(3), np.zeros(3), scale=0.06, color='royalblue', label='RGB')

    # ── HSI camera using calibrated extrinsics ────────────────────────────────
    # t_h2c is HSI origin in RGB frame (already computed above)
    # R_h2c columns are HSI axes expressed in RGB frame
    hsi_R_cam2world = R_h2c          # RGB-frame columns of HSI axes = cam→world rotation
    hsi_pos         = t_h2c.ravel()
    _frustum(ax, hsi_R_cam2world, hsi_pos, scale=0.06, color='darkorange', label='HSI')

    # ── HSI scan plane (X_h=0, shown as a large translucent strip) ───────────
    # In the RGB frame, the HSI slit plane normal is n = R_h2c @ [1,0,0] = first column of R_h2c
    # Draw the plane as a quad ±0.15 m along Y_h and ±0.2 m along Z_h
    y_ax = R_h2c @ np.array([0., 1., 0.])  # Y_h in RGB frame
    z_ax = R_h2c @ np.array([0., 0., 1.])  # Z_h in RGB frame
    plane_corners = np.array([
        hsi_pos + y_ax * 0.15 + z_ax * 0.05,
        hsi_pos - y_ax * 0.15 + z_ax * 0.05,
        hsi_pos - y_ax * 0.15 + z_ax * 0.40,
        hsi_pos + y_ax * 0.15 + z_ax * 0.40,
    ])
    ax.add_collection3d(Poly3DCollection([plane_corners.tolist()],
                                         alpha=0.15, facecolor='yellow',
                                         edgecolor='gold', lw=1.5))

    # ── Board poses ──────────────────────────────────────────────────────────
    cmap   = plt.cm.Set2
    boards = [(det["view_id"],
               np.array(det["R"]),
               np.array(det["T"]).ravel())
              for det in detections if det.get("pose_found")]

    for idx, (vid, R_b, T_b) in enumerate(boards):
        color = cmap(idx % 8)
        # Shift the label slightly so they don't overlap as much
        # e.g., offset along the board's normal vector (-Z in board frame)
        offset = R_b @ np.array([0, 0, -0.05])
        _board(ax, R_b, T_b, color=color, label=None)
        
        # Add a small dot and label at the offset position
        ctr = (R_b @ np.array([[0.12, 0.06, 0]]).T).ravel() + T_b
        lbl_pos = ctr + offset
        ax.plot([ctr[0], lbl_pos[0]], [ctr[1], lbl_pos[1]], [ctr[2], lbl_pos[2]], 
                color='gray', linestyle=':', lw=0.5)
        ax.text(*lbl_pos, vid.replace('view_', 'V'), fontsize=10, 
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    # ── Annotations and Styling ───────────────────────────────────────────────
    # Axis limits
    all_pts = np.vstack([[np.zeros(3), hsi_pos]] +
                        [T_b.reshape(1,3) for _,_,T_b in boards])
    rng = np.ptp(all_pts, axis=0).max() / 2.0 + 0.05
    mid = all_pts.mean(axis=0)
    ax.set_xlim(mid[0]-rng, mid[0]+rng)
    ax.set_ylim(mid[1]-rng, mid[1]+rng)
    ax.set_zlim(mid[2]-rng, mid[2]+rng)

    # Conference-clean styling
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    # Z-axis label is sometimes awkward in 3D, keeping it simple
    ax.set_zlabel('Z (m)', fontsize=12)
    
    # Remove gray panes for a cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Set view angle (optional, adjust as needed)
    ax.view_init(elev=20, azim=45)

    out_3d = args.out / "rig_3d.png"
    fig.savefig(str(out_3d), dpi=300, bbox_inches='tight')  # Higher DPI for paper
    print(f"  Saved 3D view -> {out_3d}")
    plt.close(fig)

if __name__ == '__main__':
    main()

