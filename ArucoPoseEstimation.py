import cv2
import numpy as np

# =====================================================================
# SECTION 1: SETTINGS
# =====================================================================

# Real side length of the printed marker in METERS (5 cm = 0.05).
# Measure the black square of your printed tag with a ruler.
MARKER_SIZE = 0.05

# The three tag types we want to detect: 4x4, 5x5 and 6x6.
# On chev.me/arucogen choose the same dictionary (e.g. "5x5 (250)").
ARUCO_DICTS = {
    "4x4": cv2.aruco.DICT_4X4_250,
    "5x5": cv2.aruco.DICT_5X5_250,
    "6x6": cv2.aruco.DICT_6X6_250,
}

# =====================================================================
# SECTION 2: OPEN THE WEBCAM AND GET ITS REAL FRAME SIZE
# =====================================================================

cap = cv2.VideoCapture(0)   # 0 = default webcam (try 1 if it does not open)

if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit

# Ask the camera for 640x480 (it may ignore this if it can't do it)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Now read the size the camera is REALLY using.
# This is how you find your camera's width and height.
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Camera resolution:", FRAME_WIDTH, "x", FRAME_HEIGHT)

# =====================================================================
# SECTION 3: CAMERA MATRIX (approximation, no calibration)
# =====================================================================
# The focal length is guessed as the image width; the center is the
# middle of the image. For better accuracy, calibrate your camera.

focal_length = FRAME_WIDTH
center_x = FRAME_WIDTH / 2
center_y = FRAME_HEIGHT / 2

camera_matrix = np.array([
    [focal_length, 0, center_x],
    [0, focal_length, center_y],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)   # assume no lens distortion

# =====================================================================
# SECTION 4: 3D SHAPE OF THE MARKER
# =====================================================================
# The marker is a flat square. Its center is the origin (0, 0, 0).
# Corner order must match OpenCV: top-left, top-right,
# bottom-right, bottom-left.

half = MARKER_SIZE / 2
obj_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)

# =====================================================================
# SECTION 5: CREATE THE DETECTORS ONCE (before the loop)
# =====================================================================
# Creating them once makes the video faster than creating them
# on every frame.

detectors = {}
for dict_name, dict_enum in ARUCO_DICTS.items():
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_enum)
    if hasattr(cv2.aruco, "ArucoDetector"):          # newer OpenCV (4.7+)
        params = cv2.aruco.DetectorParameters()
        detectors[dict_name] = cv2.aruco.ArucoDetector(aruco_dict, params)
    else:                                            # older OpenCV
        params = cv2.aruco.DetectorParameters_create()
        detectors[dict_name] = (aruco_dict, params)

print("Camera running. Hold up a 4x4, 5x5, or 6x6 ArUco tag. Press 'q' to exit.")

# =====================================================================
# SECTION 6: MAIN LOOP (runs for every video frame)
# =====================================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detection works on a black & white (gray) image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- 6a: Look for each tag type (4x4, 5x5, 6x6) ----
    for dict_name in ARUCO_DICTS:

        if hasattr(cv2.aruco, "ArucoDetector"):
            corners, ids, _ = detectors[dict_name].detectMarkers(gray)
        else:
            aruco_dict, params = detectors[dict_name]
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        # Nothing found for this type -> go to the next type
        if ids is None:
            continue

        # Draw green outline and ID number around each detected marker
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Make ids a simple flat list like [3, 7, 12] so ids_list[i] is
        # always one number (different OpenCV versions return different shapes).
        ids_list = ids.flatten()

        # ---- 6b: Estimate pose of every detected marker ----
        for i in range(len(ids_list)):
            image_points = corners[i].reshape(4, 2).astype(np.float32)

            # solvePnP compares the real 3D square with the 4 corners seen
            # in the image and returns rotation (rvec) and position (tvec).
            success, rvec, tvec = cv2.solvePnP(
                obj_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success:
                continue

            # ---- 6c: Draw 3D axes (Red = X, Green = Y, Blue = Z) ----
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE * 0.7)

            # ---- 6d: Position and depth, converted to centimeters ----
            x_cm = tvec[0][0] * 100      # left / right
            y_cm = tvec[1][0] * 100      # up / down
            depth_cm = tvec[2][0] * 100  # distance from the camera (depth)

            # ---- 6e: Orientation as angles in degrees ----
            # Convert the rotation vector to a rotation matrix, then to angles.
            R, _ = cv2.Rodrigues(rvec)
            roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

            # ---- 6f: Write the info on the video ----
            # Put the text near the marker's top-left corner
            tx = int(image_points[0][0])
            ty = int(image_points[0][1])

            lines = [
                f"{dict_name} ID:{ids_list[i]}",
                f"Depth: {depth_cm:.1f} cm",
                f"X:{x_cm:.1f} Y:{y_cm:.1f} cm",
                f"Roll:{roll:.0f} Pitch:{pitch:.0f} Yaw:{yaw:.0f}",
            ]
            for n, text in enumerate(lines):
                # each line is drawn 20 pixels above the previous one
                y_pos = ty - 10 - 20 * (len(lines) - 1 - n)
                cv2.putText(frame, text, (tx, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ---- 6g: Show the video window ----
    cv2.imshow("ArUco Pose & Depth Estimation", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================================================================
# SECTION 7: CLEAN UP
# =====================================================================
cap.release()
cv2.destroyAllWindows()