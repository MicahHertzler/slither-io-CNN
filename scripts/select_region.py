import cv2
import numpy as np
from mss import mss

with mss() as screen:
    monitor = screen.monitors[1]  # first physical monitor
    screenshot = np.asarray(screen.grab(monitor))
    image = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    x, y, width, height = cv2.selectROI(
        "Drag over the Slither.io play area, then press Enter",
        image,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyAllWindows()

    if width == 0 or height == 0:
        print("No region selected.")
    else:
        region = {
            "left": int(monitor["left"] + x),
            "top": int(monitor["top"] + y),
            "width": int(width),
            "height": int(height),
        }
        print("Paste this into browser.py:")
        print(f"REGION = {region}")