# Calibration Targets

This directory contains calibration target assets and instructions.

## Files

- `dimensions.yaml` - Physical dimensions of the calibration target
- `charuco_6x9.svg` - SVG template for the ChArUco board (placeholder)

## Generating Calibration Targets

### Option 1: Use OpenCV (Recommended)

Generate a ChArUco board using OpenCV:

```python
import cv2
from cv2 import aruco

# Create dictionary and board
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((9, 6), 0.030, 0.0225, dictionary)

# Generate image
img = board.generateImage((2700, 1800))  # 10 pixels per mm at 100 DPI

# Save as PNG
cv2.imwrite('charuco_6x9.png', img)
```

### Option 2: Online Generators

- [calib.io](https://calib.io/pages/camera-calibration-pattern-generator)
- [OpenCV ChArUco generator docs](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)

### Option 3: Use the Included Script

```bash
python -c "
import cv2
from cv2 import aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((9, 6), 0.030, 0.0225, dictionary)
img = board.generateImage((2700, 1800))
cv2.imwrite('charuco_6x9.png', img)
print('Generated charuco_6x9.png')
"
```

## Printing Guidelines

1. **Use 100% scale** - Do not let the printer resize the image
2. **Measure after printing** - Verify actual square size matches `dimensions.yaml`
3. **Use matte paper** - Glossy paper causes reflections
4. **Mount on rigid backing** - Foam board or aluminum composite
5. **Ensure planarity** - Target must be flat, no warping

## Updating dimensions.yaml

After printing and measuring, update `dimensions.yaml` with the actual measured dimensions:

```yaml
target:
  square_size_mm: 29.85  # Actual measured value
  marker_size_mm: 22.39  # Actual measured value
```

A 1% error in dimensions causes 1% error in distance measurements.

## Binary PDFs

Binary PDF files are not included in this repository. Generate them using the methods above, or contact your equipment vendor for pre-made targets.
