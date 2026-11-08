#screenshot debug view
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from mss import mss

# Update these to cover the game viewport only.
REGION = {"left": 100, "top": 80, "width": 1280, "height": 720}

CROP_SIZE = 400  # width and height of the center crop
SAVE_DIR = Path("logs/debug_frames")


def detect_cues(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Keep saturated, reasonably bright pixels.
    # Tune these thresholds after looking at your screenshots.
    lower = np.array([0, 45, 50], dtype=np.uint8)
    upper = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    annotated = frame.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Draw each detected color region so you can inspect the mask.
        cv2.drawContours(annotated, [contour], -1, (255, 200, 0), 1)

        x, y, w, h = cv2.boundingRect(contour)

        # Small, roughly round regions might be pellets.
        if area <= 120 and circularity >= 0.60:
            cv2.circle(annotated, (x + w // 2, y + h // 2), 10, (0, 255, 255), 2)
            cv2.putText(
                annotated, "pellet?", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
            )

        # Large colored regions might be snake bodies, but can include other things.
        elif area > 120:
            cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
            cv2.putText(
                annotated, "snake/blob?", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )

    return annotated, mask

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    with mss() as screen:
        while True:
            # Capture the configured screen region.
            shot = np.asarray(screen.grab(REGION))
            frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2

            # Draw a crosshair at the assumed player-head center.
            marked, mask = detect_cues(frame)
            cv2.drawMarker(
                marked,
                (center_x, center_y),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=32,
                thickness=2,
            )
            cv2.circle(marked, (center_x, center_y), 10, (0, 255, 255), 2)

            # Crop a square around the center, clipping at the image edges.
            half = CROP_SIZE // 2
            x1 = max(0, center_x - half)
            y1 = max(0, center_y - half)
            x2 = min(width, center_x + half)
            y2 = min(height, center_y + half)
            center_crop = marked[y1:y2, x1:x2]

            cv2.imshow("Full game view (q to quit, s to save)", marked)
            cv2.imshow("Center crop", center_crop)
            cv2.imshow("Color mask", mask)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(SAVE_DIR / f"{stamp}_full.png"), marked)
                cv2.imwrite(str(SAVE_DIR / f"{stamp}_center_crop.png"), center_crop)
                cv2.imwrite(str(SAVE_DIR / f"{stamp}_mask.png"), mask)
                print(f"Saved capture {stamp} in {SAVE_DIR}") 
            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()