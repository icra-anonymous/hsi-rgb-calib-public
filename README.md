



# HSI-RGB Calibration

A clean, reproducible pipeline for calibrating a rigid pushbroom hyperspectral (HSI) line-scan camera with an RGB frame camera (e.g., OAK-D).


https://github.com/user-attachments/assets/ac6191eb-0fb2-4a0c-b954-8bc1c9504b59



https://github.com/user-attachments/assets/add69f9b-e02b-4e5c-97a7-3b67ec970af6


This repository implements the Li-Wen-Qiu method for line-scan + frame camera calibration using a custom ArUco-augmented target, centering around a two-phase script workflow.

## Overview

The calibration process is split into two phases:

1. **Phase 1: Detection & Visualization** (scripts/detect_and_visualize.py)
   - Automatically detects ArUco markers in the RGB images to estimate the calibration board's 3D pose.
   - Provides an interactive UI to manually select the HSI pushbroom slit crossings on the target.
   - Saves all detected points and 3D poses to detections.json.

2. **Phase 2: Extrinsics Estimation** (scripts/run_calibration.py)
   - Reads the detections.json file.
   - Uses a non-linear optimizer to estimate the spatial transformation (rotation and translation) between the RGB camera and the HSI line-scan camera.
   - Generates visual projections of the estimated HSI slit onto the RGB images to verify the calibration quality.

## Installation

`ash
# Clone the repository
git clone <your-repo-url>
cd hsi-rgb-calib-public

# Install as a package (installs dependencies like OpenCV, Matplotlib, etc.)
pip install -e .
`

## Usage Example

An example dataset is provided in data/example_session/.

### Phase 1: Detect and Visualize

Run the interactive detection script on the example dataset:

`ash
python scripts/detect_and_visualize.py \
    --rgb-dir data/example_session \
    --calib-json data/example_session/calibration.json \
    --out output/phase1
`

An interactive window will pop up for each image pair, prompting you to drag the horizontal and diagonal edge lines to the correct locations on the HSI profile. 

### Phase 2: Run Calibration

Once Phase 1 is complete and output/phase1/detections.json is generated, run the estimation script:

`ash
python scripts/run_calibration.py \
    --detections output/phase1/detections.json \
    --calib-json data/example_session/calibration.json \
    --out output/phase2
`

*(Optional) You can use the --physical-init flag to provide a physical prior for the optimizer if it struggles to converge from a generic closed-form initialization.*

## Outputs

After Phase 2 finishes, the output/phase2 directory will contain:
- hsi_rgb_extrinsics.json: The final 3D rotation and translation between the cameras, plus estimated HSI focal length and principal point.
- Reprojection Visualizations: Images projecting the calibrated HSI slit back onto the RGB frames to help you verify alignment visually.
- A 3D rig visualization plot showing the solved poses of the board and cameras.

## License
MIT License
