#temporary debugging tool, so it doesn’t replace browser.py, train.py, or another project file.

import opencv-python

import cv2
import numpy as np
from mss import mss

# Set these to the game viewport's screen coordinates.
# Example: left edge 100px, top edge 80px, size 1280x720.
REGION = {"left": 100, "top": 80, "width": 1280, "height": 720}

with mss() as screen:
    while True:
        screenshot = np.array(screen.grab(REGION))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        height, width = frame.shape[:2]
        center = (width // 2, height // 2)

        # Draw the assumed head-center marker.
        cv2.drawMarker(
            frame, center, (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=30,
            thickness=2,
        )
        cv2.circle(frame, center, 8, (0, 255, 255), 2)

        cv2.imshow("Game center check", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            cv2.imwrite("center_check.png", frame)
            print("Saved center_check.png")

        if key == ord("q"):
            break

cv2.destroyAllWindows()