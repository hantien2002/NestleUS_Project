import cv2
import json
import numpy as np

points = []
scale = 1.0  # same with pipeline

def mouse_callback(event, x, y, flags, param):
    global points, scale

    if event == cv2.EVENT_LBUTTONDOWN:
        # save original xy
        orig_x = x / scale
        orig_y = y / scale
        points.append((orig_x, orig_y))


def main(video_path):
    global points, scale

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to read video")
        return

    # same with pipeline
    h, w = frame.shape[:2]
    scale = 1000 / max(w, h)

    display_w = int(w * scale)
    display_h = int(h * scale)

    display_frame = cv2.resize(frame, (display_w, display_h))
    clone = display_frame.copy()

    # Window
    cv2.namedWindow("Draw Zone", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Draw Zone", display_w, display_h)
    cv2.setMouseCallback("Draw Zone", mouse_callback)

    while True:
        img = clone.copy()

        # ===== original xy → display xy =====
        display_points = [(int(x * scale), int(y * scale)) for x, y in points]

        # draw points
        for p in display_points:
            cv2.circle(img, p, 5, (0, 255, 0), -1)

        # draw lines
        if len(display_points) >= 2:
            pts = np.array(display_points, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        # fill
        if len(display_points) >= 3:
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (180, 220, 255))
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

        cv2.imshow("Draw Zone", img)

        key = cv2.waitKey(1) & 0xFF

        # ===== save and exit =====
        if key == ord('s'):
            zone_data = [
                {
                    "id": "zone_a",
                    "name": "zone_a",
                    "points": [[int(x), int(y)] for x, y in points],  # original coordinates
                    "color": [180, 220, 255]
                }
            ]
            with open("zones.json", "w") as f:
                json.dump(zone_data, f)

            print("✅ Saved zones.json and exit")
            break

        # clear
        if key == ord('c'):
            points.clear()

        # exit
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("video/10_16_2.mp4")