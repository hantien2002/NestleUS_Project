import os
import time
import glob
import cv2
import json
import argparse
from pathlib import Path

from ppe_pipeline.detection import load_model, boxes_overlapping_region
from ppe_pipeline.zones import ZoneRegistry, zone_for_bbox
from ppe_pipeline.tracking_alerting import TriggerConfig, ViolationTracker
from ppe_pipeline.viz import FrameAnnotator

def process_stream(input_dir: str, model_path: str, zones_path: str, fps: float = 30.0):
    # 1. Initialize Components
    model = load_model(model_path)
    model.to("cuda")
    with open(zones_path, "r") as f:
        registry = ZoneRegistry(json.load(f))
        
    # window_size=60, threshold=30 exactly matches your "more than half of 60 frames" logic
    cfg = TriggerConfig(window_size_frames=60, violation_threshold_frames=30)
    tracker = ViolationTracker(config=cfg) 
    annotator = FrameAnnotator()

    # Model Class IDs
    names = getattr(model, "names", {})
    person_ids = [i for i, n in names.items() if str(n).lower() == "person"]
    helmet_ids = [i for i, n in names.items() if str(n).lower() == "helmet"]
    vest_ids = [i for i, n in names.items() if str(n).lower() == "vest"]
    target_classes = sorted(set(person_ids + helmet_ids + vest_ids))

    processed_files = set()
    frame_id = 0
    last_log_count = 0  # To track newly generated alerts

    print(f"Watching '{input_dir}' for new frames...")
    print("Press 'q' in the video window to stop the stream.")
    print("-" * 60)

    try:
        while True:
            # Grab sorted frames
            current_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg"))) 
            new_files = [f for f in current_files if f not in processed_files]

            for img_path in new_files:
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                time_sec = frame_id / fps

                # 2. Stateful Tracking Prediction
                results = model.track(
                    source=frame, 
                    tracker="bytetrack.yaml", 
                    persist=True, 
                    classes=target_classes, 
                    conf=0.35, 
                    iou=0.55, 
                    verbose=False
                )

                persons_data = []
                r = results[0]
                boxes = getattr(r, "boxes", None)
                
                # 3. PPE Extraction Logic
                if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy().astype(float).tolist()
                    cls = boxes.cls.cpu().numpy().astype(int).tolist() if getattr(boxes,"cls",None) is not None else [None]*len(xyxy)
                    ids = boxes.id.cpu().numpy().astype(int).tolist() if getattr(boxes,"id",None) is not None else [None]*len(xyxy)
                    
                    person_by_tid = {}
                    helmet_boxes, vest_boxes = [], []
                    
                    for b, c, tid in zip(xyxy, cls, ids):
                        if c in person_ids and tid is not None:
                            person_by_tid[int(tid)] = b
                        elif c in helmet_ids:
                            helmet_boxes.append(b)
                        elif c in vest_ids:
                            vest_boxes.append(b)

                    for tid, pbox in person_by_tid.items():
                        x1, y1, x2, y2 = map(float, pbox)
                        h = y2 - y1
                        head_region = (x1, y1, x2, y1 + 0.30 * h)
                        torso_region = (x1, y1 + 0.30 * h, x2, y1 + 0.80 * h)
                        
                        mhelm = boxes_overlapping_region(helmet_boxes, head_region, 0.15)
                        mvest = boxes_overlapping_region(vest_boxes, torso_region, 0.15)
                        
                        zname, zid, (u, v) = zone_for_bbox(pbox, registry, "bottom_center")

                        persons_data.append({
                            "id": int(tid),
                            "person_xyxy": [x1, y1, x2, y2],
                            "helmet_xyxy": mhelm,
                            "vest_xyxy": mvest,
                            "has_helmet": len(mhelm) > 0,
                            "has_vest": len(mvest) > 0,
                            "zone_name": zname,
                            "zone_id": zid,
                            "zone_point_uv": [float(u), float(v)],
                        })

                # 4. Update Alerting State Machine
                tracker.process_frame(time_sec, persons_data)

                # --- NEW: Check for and print new alerts ---
                current_log_count = len(tracker.second_logs)
                if current_log_count > last_log_count:
                    # Iterate over only the new logs generated in this frame/second
                    for log in tracker.second_logs[last_log_count:]:
                        print(f"[🚨 ALERT] Time: {log['second']}s | Track ID: {log['track_id']} | "
                              f"Zone: {log['zone_id']} | Type: {log['violation_type']} | "
                              f"History Ratio: {log['trigger_ratio']}")
                    last_log_count = current_log_count

                # 5. Real-Time Visualization
                annotator.draw_zones(frame, registry)
                for p in persons_data:
                    annotator.draw_person(frame, p)

                # --- NEW: Show live video stream ---
                # Resize if your frames are massive (e.g., 4K) so it fits on screen
                # display_frame = cv2.resize(frame, (1280, 720)) 
                h, w = frame.shape[:2]
                scale = 1000 / max(w, h)   # 控制最长边
                display_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                cv2.imshow("Live PPE Detection Stream", display_frame)
                
                # waitKey(1) renders the frame and checks for keypress. 
                # If processing is faster than 30fps, you might want to increase this 
                # to waitKey(int(1000/fps)) to artificially slow it down to 1x speed.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stream stopped by user.")
                    cv2.destroyAllWindows()
                    return

                processed_files.add(img_path)
                frame_id += 1

            # Brief sleep to prevent CPU pegging while waiting for new frames to arrive in the folder
            time.sleep(0.05) 

    except KeyboardInterrupt:
        print("\nStopping stream runner...")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder receiving stream frames")
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--zones", default="zones.json")
    args = parser.parse_args()

    process_stream(args.input_dir, args.model, args.zones)