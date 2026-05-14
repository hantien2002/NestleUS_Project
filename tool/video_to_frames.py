import cv2, os, argparse, json

def video_to_frames(video_path, out_dir, img_ext=".jpg"):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}{img_ext}")
        cv2.imwrite(out_path, frame)
        frame_idx += 1

    cap.release()

    meta = {
        "video_path": video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "num_frames": frame_idx,
        "pattern": f"frame_%06d{img_ext}"
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {frame_idx} frames to {out_dir}")
    print(f"FPS: {fps}, size: {width}x{height}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ext", default=".jpg")
    args = parser.parse_args()
    video_to_frames(args.video, args.out, args.ext)