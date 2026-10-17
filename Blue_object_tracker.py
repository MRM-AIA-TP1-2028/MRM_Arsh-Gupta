import cv2
import numpy as np
from collections import deque
from math import hypot

# Overall idea: read camera frames, find objects by COLOUR, measure them,
# draw the results on the frame, repeat until 'q' is pressed.

# Configurations

cap = cv2.VideoCapture(0)   # 0 = default camera (try 1 if it does not open)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Blue object HSV range
# HSV = Hue (colour), Saturation (how vivid), Value (brightness).
# OpenCV ranges: H 0-179, S 0-255, V 0-255. Blue hue is about 120.
# Format: [H, S, V]
BLUE_LOWER = np.array([90, 70, 40])
BLUE_UPPER = np.array([135, 255, 255])

# tennis ball HSV range (yellow-green hue is about 30-45)
BALL_GREEN_LOWER = np.array([20, 40, 40])
BALL_GREEN_UPPER = np.array([50, 255, 255])

# Second range in case the camera shows the ball with a pink tint
BALL_PINK_LOWER = np.array([145, 40, 40])
BALL_PINK_UPPER = np.array([179, 255, 255])

# Blobs smaller than this (in px^2) are treated as noise and ignored
MIN_BLUE_AREA = 300
MIN_BALL_AREA = 500

# Store the recent blue-object centre positions.
# maxlen=50 means only the last 50 points are kept; older ones drop off
# automatically, so the trail line has a limited length.
blue_trail = deque(maxlen=50)

#camera loop part
# Everything inside this loop runs once for every camera frame.

while True:
    # ret = True if a frame was read, frame = the image (BGR colour order)
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read camera frame.")
        break

    # Smooth the image to reduce noise (smoother masks later)
    blurred = cv2.GaussianBlur(frame, (7,7), 0)

    # Convert BGR image to HSV colour space.
    # HSV is used because hue stays similar when lighting changes,
    # while BGR values change a lot.
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # ---------------- Blue object detection ----------------

    # Mask = black and white image: white where the pixel is inside the
    # blue range, black everywhere else.
    blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    # Clean the mask using a 5x5 square "kernel":
    #   OPEN  removes small white specks (noise)
    #   CLOSE fills small black holes inside the object
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(
        blue_mask, cv2.MORPH_OPEN, kernel
    )
    blue_mask = cv2.morphologyEx(
        blue_mask, cv2.MORPH_CLOSE, kernel
    )

    # Contours = outlines of the white blobs in the mask.
    # Each contour is a list of points along the edge of one blob.
    blue_contours, _ = cv2.findContours(
        blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Start each frame assuming nothing was found
    blue_center = None

    # Choose the largest blue region above the minimum area
    valid_blue = [
        c for c in blue_contours
        if cv2.contourArea(c) >= MIN_BLUE_AREA
    ]

    if valid_blue:
        blue_contour = max(valid_blue, key=cv2.contourArea)

        # Area in pixels squared (pixels inside the outline,
        # so it is correct for any shape)
        blue_area = cv2.contourArea(blue_contour)

        # Bounding rectangle works for any shape
        # x, y = top-left corner, w = width, h = height
        x, y, w, h = cv2.boundingRect(blue_contour)

        # Centre of the bounding box
        blue_center = (x + w // 2, y + h // 2)
        blue_trail.append(blue_center)

        # Draw the contour and bounding box
        # (colours are BGR, so (255, 0, 0) is blue)
        cv2.drawContours(
            frame, [blue_contour], -1, (255, 0, 0), 2
        )
        cv2.rectangle(
            frame, (x, y), (x + w, y + h), (255, 0, 0), 2
        )

        # Filled dot at the centre (thickness -1 = filled)
        cv2.circle(frame, blue_center, 5, (255, 0, 0), -1)

        # max(..., 20) keeps the text on screen if the object is at the top
        cv2.putText(
            frame, f"Blue Area: {blue_area:.0f} px^2",
            (x, max(y - 35, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2
        )

        cv2.putText(
            frame, f"Box: {w} x {h} px",
            (x, max(y - 12, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2
        )

        # Draw the blue object's movement trail:
        # join each saved centre to the next one with a short line
        for i in range(1, len(blue_trail)):
            cv2.line(
                frame, blue_trail[i - 1], blue_trail[i],
                (255, 0, 0), 2
            )

    else:
        # Do not connect a new detection to an old trail
        blue_trail.clear()

    # ---------------- Detection of tennis ball ----------------

    # Make one mask per colour range, then merge them with OR
    # (a pixel is white if it is in EITHER range)
    green_mask = cv2.inRange(hsv, BALL_GREEN_LOWER, BALL_GREEN_UPPER)
    pink_mask = cv2.inRange(hsv,BALL_PINK_LOWER,BALL_PINK_UPPER)
    ball_mask = cv2.bitwise_or(green_mask,pink_mask)

    # Same cleaning as for the blue mask
    ball_mask = cv2.morphologyEx(
        ball_mask, cv2.MORPH_OPEN, kernel
    )
    ball_mask = cv2.morphologyEx(
        ball_mask, cv2.MORPH_CLOSE, kernel
    )

    ball_contours, _ = cv2.findContours(
        ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    ball_center = None

    # Check candidate contours for a roughly circular shape.
    # Largest first, and we stop at the first one that passes (see break).
    for contour in sorted(
        ball_contours, key=cv2.contourArea, reverse=True
    ):
        ball_area = cv2.contourArea(contour)

        # Too small to be the ball
        if ball_area < MIN_BALL_AREA:
            continue

        # Length of the outline (True = closed shape)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:   # avoid dividing by zero
            continue

        # Circularity = 4*pi*area / perimeter^2
        # 1.0 = perfect circle, square is about 0.78, thin shapes are near 0
        circularity = (
            4 * np.pi * ball_area / (perimeter * perimeter)
        )

        # Smallest circle that contains the blob:
        # gives the centre (cx, cy) and the radius in pixels
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        # Tune these limits for your camera and tennis ball
        # Accept only if it is round enough AND a sensible size
        if (
              circularity > 0.75
              and radius >= 10
            and radius <= 150
        ):
            # Drawing functions need whole numbers
            ball_center = (int(cx), int(cy))
            ball_radius = int(radius)

            # Draw ball circle and centre
            cv2.circle(
                frame, ball_center, ball_radius, (0, 255, 255), 2
            )
            cv2.circle(
                frame, ball_center, 4, (0, 0, 255), -1
            )

            cv2.putText(
                frame, f"Ball Area: {ball_area:.0f} px^2",
                (ball_center[0] + 10, ball_center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )

            cv2.putText(
                frame,
                f"Center: {ball_center}  Radius: {radius:.1f} px",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
            )

            # Ball found, no need to check smaller blobs
            break

    # ---------------- Relative positioning ----------------
    # Only possible when BOTH objects were found in this frame

    if blue_center is not None and ball_center is not None:
        # Draw a line from blue centre to tennis ball centre
        cv2.line(
            frame, blue_center, ball_center, (0, 255, 0), 2
        )

        # Difference in position (blue minus ball)
        # dx > 0 -> blue is to the right of the ball
        # dy > 0 -> blue is below the ball (in images, y grows downward)
        dx = blue_center[0] - ball_center[0]
        dy = blue_center[1] - ball_center[1]

        # Straight-line distance between the two centres, in pixels
        distance = hypot(dx, dy)

        # Convert the numbers into words
        if dx < 0:
            horizontal = "Left"
        elif dx > 0:
            horizontal = "Right"
        else:
            horizontal = "Same X"

        if dy < 0:
            vertical = "Above"
        elif dy > 0:
            vertical = "Below"
        else:
            vertical = "Same Y"

        cv2.putText(
            frame,
            f"Blue is {horizontal}, {vertical}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"Delta X: {dx}  Delta Y: {dy}  Distance: {distance:.1f}px",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )

    elif blue_center is not None:
        # Blue is visible but the ball is not
        cv2.putText(
            frame, "Tennis ball not detected",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 165, 255), 2
        )

    # Display part
    # The two mask windows are for debugging: white = detected colour

    cv2.imshow("Blue Object and Tennis Ball Tracker", frame)
    cv2.imshow("Blue Mask", blue_mask)
    cv2.imshow("Tennis Ball Mask", ball_mask)

    cv2.moveWindow("Blue Object and Tennis Ball Tracker", 100, 50)
    cv2.moveWindow("Blue Mask", 700, 50)
    cv2.moveWindow("Tennis Ball Mask", 700, 40)

    # Press q to quit.
    # waitKey(1) also lets the windows refresh, so it must stay in the loop.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()