# Blue Object and Tennis Ball Tracker

This program reads a live camera stream with OpenCV and detects a blue object and a tennis ball in every frame. The two objects are treated separately. The blue object can have any shape, so no circular form is assumed for it.

## Output on the screen

| Object | Information displayed |
|---|---|
| Blue object | Contour, bounding box, area in px², box width and height, a line that follows its centre, and its position relative to the tennis ball |
| Tennis ball | Enclosing circle, centre, radius in px and area in px² |
| Both objects | A line between the two centres, the horizontal and vertical offset in px, and the distance in px |

Three windows open when the program starts: the annotated camera image, the blue mask and the tennis ball mask. The two mask windows show which pixels were accepted as each colour, and they are useful for adjusting the colour limits.

## Requirements and usage

The program requires Python 3 together with the `opencv-python` and `numpy` packages.

```bash
pip install opencv-python numpy
python tracker.py
```

Press `q` while the camera window is selected to close the program. If the camera does not open, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.

## Approach

The task contains two detection problems with different conditions. The tennis ball has a known colour and a round shape. The blue object has a known colour only, because its shape can vary. For this reason, the program uses one pipeline for each object and combines the results only at the end, when the relative position is calculated.

Colour was chosen as the main cue because it is fast to compute and works on an ordinary camera without a trained model. The image is converted from BGR to HSV before any colour test. Hue describes the colour itself, whereas saturation and value describe its intensity and brightness. A change in lighting therefore alters the saturation and value more than the hue, and a single hue range remains usable under different conditions.

Shape is used only for the ball. A blob that has the colour of the ball is accepted if it is round enough and has a reasonable size. The blue object is measured with the contour area and the bounding rectangle, and both measures are valid for any shape.

## Method

Each frame passes through the following steps.

1. **Capture.** A frame is read from the camera with `cv2.VideoCapture`.
2. **Smoothing.** A 7x7 Gaussian blur reduces sensor noise, so the colour masks contain fewer isolated pixels.
3. **Colour conversion.** The frame is converted from BGR to HSV with `cv2.cvtColor`.
4. **Colour masks.** The function `cv2.inRange` produces a binary mask for each colour. The blue range is H 90 to 135. The ball uses a yellow-green range (H 20 to 50) and a second range for pink tones (H 145 to 179), and the two masks are merged with a bitwise OR.
5. **Mask cleaning.** A morphological opening removes small isolated specks, and a closing fills small holes inside an object. Both operations use a 5x5 kernel.
6. **Contours.** The function `cv2.findContours` returns the outline of each white region in the mask.
7. **Blue object measurements.** Regions smaller than 300 px² are discarded. The largest remaining region is selected. Its area comes from `cv2.contourArea`, and its box comes from `cv2.boundingRect`. The centre is the middle point of the box.
8. **Centre trail.** The last 50 centre points are stored in a `deque`. Lines drawn between consecutive points form the tracking line. The trail is cleared when the object disappears, which prevents a false line between two separate detections.
9. **Ball measurements.** Candidate regions are examined from largest to smallest. A region below 500 px² is skipped. For the others, circularity is calculated as 4πA / P², where A is the area and P is the perimeter. A perfect circle gives a value of 1.0, and a square gives about 0.78. A region is accepted as the ball if circularity is above 0.75 and the radius lies between 10 and 150 px. The function `cv2.minEnclosingCircle` then gives the centre and the radius.
10. **Relative position.** When both objects are visible, the program calculates dx and dy as the blue centre minus the ball centre. A positive dx means that the blue object lies to the right of the ball. A positive dy means that it lies below the ball, because the y axis of an image points downward. The distance is calculated with the Euclidean formula.
11. **Display.** The results are drawn on the frame, and the three windows are updated.

## Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `BLUE_LOWER`, `BLUE_UPPER` | [90, 70, 40], [135, 255, 255] | HSV range of the blue object |
| `BALL_GREEN_LOWER`, `BALL_GREEN_UPPER` | [20, 40, 40], [50, 255, 255] | HSV range of the ball |
| `BALL_PINK_LOWER`, `BALL_PINK_UPPER` | [145, 40, 40], [179, 255, 255] | Second HSV range for a pink tint |
| `MIN_BLUE_AREA` | 300 px² | Smallest accepted blue region |
| `MIN_BALL_AREA` | 500 px² | Smallest accepted ball region |
| Circularity limit | 0.75 | Minimum roundness of the ball |
| Trail length | 50 points | Number of centre positions stored |

These values depend on the camera and on the lighting. The mask windows show the effect of a change immediately.

## Other possible methods

- **Hough circle transform.** The function `cv2.HoughCircles` detects circles from edges, so no colour information is needed. It is sensitive to its parameters and may report false circles in a cluttered scene.
- **CamShift or MeanShift.** The user selects an object, and the program follows the colour histogram of that selection. This method adapts to changes in the size and orientation of the object.
- **OpenCV trackers (KCF, CSRT).** These trackers follow a selected region from frame to frame and run quickly. They do not find the object again after it is lost.
- **Kalman filter.** A filter applied to the centre position gives a smoother trail and can estimate the position during a short occlusion.
- **Deep learning detectors (YOLO, SSD).** These models are the most robust to lighting and background, and some include a sports ball class. They need a trained model and more computing power, and the blue object would require a custom class or a segmentation model.
- **Automatic colour calibration.** The colour limits can be computed from a sample region at start-up, which removes the need to set them by hand.

## Limitations

- Only the largest blue region is tracked. A second blue object in the frame is ignored.
- The colour ranges for the ball are wide. A round object of similar colour may be accepted as the ball, and the pink range may also match some red objects.
- Large changes in lighting or camera white balance require new colour limits.
- All distances and areas are in pixels. Conversion to physical units requires a camera calibration.
