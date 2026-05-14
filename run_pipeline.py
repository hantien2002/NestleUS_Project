import argparse, json, os
from ppe_pipeline.detection import load_model, detect_to_json
from ppe_pipeline.viz import render_video_from_json
from ppe_pipeline.tracking_alerting import run_pipeline_per_second, TriggerConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--zones", default="zones.json")
    parser.add_argument("--outdir", default="result")
    args = parser.parse_args()

    with open(args.zones, "r") as f:
        zones = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.input))[0]
    json_path = os.path.join(args.outdir, f"{base}_detections.json")
    annotated_path = os.path.join(args.outdir, f"{base}_annotated.mp4")
    alert_dir = os.path.join(args.outdir, "alerting_outputs")

    model = load_model(args.model)
    detect_to_json(args.input, model, zones, json_path=json_path)
    render_video_from_json(json_path, args.input, annotated_path, zones)

    cfg = TriggerConfig(window_size_frames=60, violation_threshold_frames=30)
    csv_path = run_pipeline_per_second(json_path, alert_dir, trigger_config=cfg)

    print("Detection JSON:", json_path)
    print("Annotated video:", annotated_path)
    print("Alert CSV:", csv_path)

if __name__ == "__main__":
    main()