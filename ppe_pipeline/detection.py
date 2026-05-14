import os, glob, cv2, numpy as np
from ultralytics import YOLO
from .io import save_json
from .zones import ZoneRegistry, zone_for_bbox

def make_output_name(video_in: str, out_dir: str="result", ext: str=".json") -> str:
    os.makedirs(out_dir, exist_ok=True)
    base=os.path.basename(video_in)
    name,_=os.path.splitext(base)
    pattern=os.path.join(out_dir, f"{name}_*{ext}")
    existing=glob.glob(pattern)
    used=[]
    for f in existing:
        try: used.append(int(os.path.splitext(f)[0].split("_")[-1]))
        except ValueError: pass
    nxt=max(used)+1 if used else 1
    return os.path.join(out_dir, f"{name}_{nxt}{ext}")

def frac_inside(box, region):
    x1,y1,x2,y2=map(float, box)
    rx1,ry1,rx2,ry2=map(float, region)
    ix1=max(x1,rx1); iy1=max(y1,ry1); ix2=min(x2,rx2); iy2=min(y2,ry2)
    iw=max(0.0, ix2-ix1); ih=max(0.0, iy2-iy1)
    inter=iw*ih
    area=max(0.0,(x2-x1))*max(0.0,(y2-y1))
    return (inter/area) if area>0 else 0.0

def boxes_overlapping_region(dets_xyxy, region, min_frac=0.15):
    out=[]
    for b in dets_xyxy:
        if frac_inside(b, region) >= float(min_frac):
            out.append(b)
    return out

def load_model(model_path: str):
    return YOLO(model_path)

def detect_to_json(video_path: str, model, zones, json_path: str=None, tracker_yaml: str="bytetrack.yaml",
                  conf: float=0.35, iou: float=0.55, zone_anchor: str="bottom_center",
                  head_frac: float=0.30, torso_top_frac: float=0.30, torso_bot_frac: float=0.80,
                  min_ppe_frac: float=0.15):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open input video: {video_path}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    names=getattr(model,"names",{})
    person_ids=[i for i,n in names.items() if str(n).lower()=="person"]
    helmet_ids=[i for i,n in names.items() if str(n).lower()=="helmet"]
    vest_ids=[i for i,n in names.items() if str(n).lower()=="vest"]
    target_classes=sorted(set(person_ids+helmet_ids+vest_ids))
    registry=ZoneRegistry(zones) if zones else None
    if json_path is None: json_path=make_output_name(video_path, ext=".json")
    frames=[]
    results=model.track(source=video_path, tracker=tracker_yaml, persist=True, stream=True,
                        classes=target_classes, conf=conf, iou=iou, verbose=False)
    frame_id=0
    for r in results:
        fr={"frame_index": frame_id, "time_sec": (frame_id/float(fps)) if fps else None, "persons": []}
        boxes=getattr(r,"boxes",None)
        if boxes is None or boxes.xyxy is None or len(boxes)==0:
            frames.append(fr); frame_id += 1; continue
        xyxy=boxes.xyxy.cpu().numpy().astype(float).tolist()
        cls=boxes.cls.cpu().numpy().astype(int).tolist() if getattr(boxes,"cls",None) is not None else [None]*len(xyxy)
        ids=boxes.id.cpu().numpy().astype(int).tolist() if getattr(boxes,"id",None) is not None else [None]*len(xyxy)
        person_by_tid={}
        helmet_boxes=[]
        vest_boxes=[]
        for b,c,tid in zip(xyxy,cls,ids):
            if c in person_ids and tid is not None:
                person_by_tid[int(tid)]=b
            elif c in helmet_ids:
                helmet_boxes.append(b)
            elif c in vest_ids:
                vest_boxes.append(b)
        for tid,pbox in person_by_tid.items():
            x1,y1,x2,y2=map(float,pbox)
            h=y2-y1
            head_region=(x1, y1, x2, y1+head_frac*h)
            torso_region=(x1, y1+torso_top_frac*h, x2, y1+torso_bot_frac*h)
            mhelm=boxes_overlapping_region(helmet_boxes, head_region, min_ppe_frac)
            mvest=boxes_overlapping_region(vest_boxes, torso_region, min_ppe_frac)
            has_helmet=len(mhelm)>0
            has_vest=len(mvest)>0
            zname,zid,(u,v)=zone_for_bbox(pbox, registry, zone_anchor) if registry else (None,None,(0.5*(x1+x2),y2))
            fr["persons"].append({
                "id": int(tid),
                "person_xyxy": [x1,y1,x2,y2],
                "helmet_xyxy": mhelm,
                "vest_xyxy": mvest,
                "has_helmet": bool(has_helmet),
                "has_vest": bool(has_vest),
                "zone_name": zname,
                "zone_id": zid,
                "zone_point_uv": [float(u), float(v)],
            })
        frames.append(fr)
        frame_id += 1
    data={"video": video_path, "fps": float(fps), "width": int(W), "height": int(H), "frames": frames}
    save_json(data, json_path, indent=2)
    return json_path

def process_single_frame(result, model, registry, zone_anchor="bottom_center",
                         head_frac=0.30, torso_top_frac=0.30,
                         torso_bot_frac=0.80, min_ppe_frac=0.15):
    names = getattr(model, "names", {})
    person_ids = [i for i, n in names.items() if str(n).lower() == "person"]
    helmet_ids = [i for i, n in names.items() if str(n).lower() == "helmet"]
    vest_ids = [i for i, n in names.items() if str(n).lower() == "vest"]

    persons_output = []
    boxes = getattr(result, "boxes", None)

    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return persons_output

    xyxy = boxes.xyxy.cpu().numpy().astype(float).tolist()
    cls = boxes.cls.cpu().numpy().astype(int).tolist() if getattr(boxes, "cls", None) is not None else [None] * len(xyxy)
    ids = boxes.id.cpu().numpy().astype(int).tolist() if getattr(boxes, "id", None) is not None else [None] * len(xyxy)

    person_by_tid = {}
    helmet_boxes = []
    vest_boxes = []

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

        head_region = (x1, y1, x2, y1 + head_frac * h)
        torso_region = (x1, y1 + torso_top_frac * h, x2, y1 + torso_bot_frac * h)

        mhelm = boxes_overlapping_region(helmet_boxes, head_region, min_ppe_frac)
        mvest = boxes_overlapping_region(vest_boxes, torso_region, min_ppe_frac)

        has_helmet = len(mhelm) > 0
        has_vest = len(mvest) > 0

        zname, zid, (u, v) = zone_for_bbox(pbox, registry, zone_anchor) if registry else (None, None, (0.5 * (x1 + x2), y2))

        persons_output.append({
            "id": int(tid),
            "person_xyxy": [x1, y1, x2, y2],
            "helmet_xyxy": mhelm,
            "vest_xyxy": mvest,
            "has_helmet": bool(has_helmet),
            "has_vest": bool(has_vest),
            "zone_name": zname,
            "zone_id": zid,
            "zone_point_uv": [float(u), float(v)],
        })

    return persons_output