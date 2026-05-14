import cv2, numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from .io import load_json
from .zones import ZoneRegistry

class FrameAnnotator:
    def __init__(self, thickness: int = 0.5):
        self.thick = thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    def draw_dashed_poly(self,img, pts, color, thickness=1, dash_length=10):
        pts = pts.reshape(-1, 2)

        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i+1) % len(pts)]

            dist = int(((x2-x1)**2 + (y2-y1)**2)**0.5)
            for j in range(0, dist, dash_length*2):
                start_ratio = j / dist
                end_ratio = min((j + dash_length) / dist, 1)

                sx = int(x1 + (x2 - x1) * start_ratio)
                sy = int(y1 + (y2 - y1) * start_ratio)
                ex = int(x1 + (x2 - x1) * end_ratio)
                ey = int(y1 + (y2 - y1) * end_ratio)

                cv2.line(img, (sx, sy), (ex, ey), color, thickness)
    def draw_zones(self, img: np.ndarray, registry: ZoneRegistry):
        for zone in registry.zones:
            overlay = img.copy()
            cv2.fillPoly(overlay, [zone.polygon], zone.color)
            alpha = 0.12  
            
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            self.draw_dashed_poly(img, zone.polygon, zone.color, 1, 10)
            zx = int(np.min(zone.polygon[:, 0])) + 5
            zy = int(np.max(zone.polygon[:, 1])) - 5
            cv2.putText(img, zone.id, (zx, zy), self.font, 1.15, zone.color, 2, cv2.LINE_AA)
    def _bbox_color(self, has_helmet: bool, has_vest: bool):
        if has_helmet and has_vest: return (0, 200, 0)      
        if has_helmet and not has_vest: return (0, 180, 220) 
        if not has_helmet and has_vest: return (200, 100, 0) 
        return (180, 0, 0)                                  
    def draw_corner_box(self,img, x1, y1, x2, y2, color, thickness=1, length=20):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)

        
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

        
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)

        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)
    def draw_person(self, img: np.ndarray, person_data: dict):
        person_xyxy = person_data.get("person_xyxy")
        if not person_xyxy:
            return

        x1, y1, x2, y2 = map(float, person_xyxy)
        tid = person_data.get("id", "?")
        has_helmet = bool(person_data.get("has_helmet", False))
        has_vest = bool(person_data.get("has_vest", False))
        zone_id = person_data.get("zone_id") or person_data.get("zone_name")

    
        box_color = self._bbox_color(has_helmet, has_vest)
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            box_color,
            -1  
        )
        alpha = 0.2  
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        self.draw_corner_box(img, x1, y1, x2, y2, box_color, 2, 10)

        
        for hb in person_data.get("helmet_xyxy", []):
            hx1, hy1, hx2, hy2 = map(float, hb)
            cv2.rectangle(img, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (255, 255, 0), 1)

        for vb in person_data.get("vest_xyxy", []):
            vx1, vy1, vx2, vy2 = map(float, vb)
            cv2.rectangle(img, (int(vx1), int(vy1)), (int(vx2), int(vy2)), (255, 0, 255),1)

        # ===== zone 点 =====
        if person_data.get("zone_point_uv") is not None:
            u, v = map(float, person_data["zone_point_uv"])
            cv2.circle(img, (int(u), int(v)), 6, (0, 255, 0), -1)

            txt = f"IN {zone_id}" if zone_id else "OUT"
            cv2.putText(
                img,
                txt,
                (int(u), int(v) + 26),
                self.font,
                0.5,
                box_color,
                1,
                cv2.LINE_AA
            )

        ppe_labels = []
        if has_helmet:
            ppe_labels.append("Helmet")
        if has_vest:
            ppe_labels.append("Vest")

        ppe_str = " ".join(ppe_labels)
        label_text = f"ID{tid}: {ppe_str}" if ppe_str else f"ID{tid}: None"

        font_scale = 0.3
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, self.font, font_scale, 1
        )

                
        font_scale = 0.5
        thickness = 1

        (label_w, label_h), baseline = cv2.getTextSize(
            label_text, self.font, font_scale, thickness
        )

        bg_x1 = int(x1)
        bg_y1 = int(y1) - label_h - 6
        bg_x2 = int(x1) + label_w + 4
        bg_y2 = int(y1)
        
        if bg_y1 < 0:
            bg_y1 = int(y1)
            bg_y2 = int(y1) + label_h + 6

        
        overlay = img.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, -1)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

        
        text_x = bg_x1 + 2
        text_y = bg_y2 - 3

        cv2.putText(
            img,
            label_text,
            (text_x, text_y),
            self.font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

def render_video_from_json(json_path: str, video_path: str, output_path: str, zones: List[Dict[str,Any]]):
    data=load_json(json_path)
    frames_map={int(fr["frame_index"]): fr for fr in data.get("frames", [])}
    registry=ZoneRegistry(zones)
    annotator=FrameAnnotator()
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Could not open video: {video_path}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS) or float(data.get("fps") or 30.0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    writer=cv2.VideoWriter(output_path, fourcc, fps, (W,H))
    frame_idx=0
    while True:
        ret, frame=cap.read()
        if not ret: break
        annotator.draw_zones(frame, registry)
        fr_data=frames_map.get(frame_idx)
        if fr_data:
            for p in fr_data.get("persons", []):
                annotator.draw_person(frame, p)
        writer.write(frame)
        frame_idx += 1
    cap.release(); writer.release()
    return output_path
