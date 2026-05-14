import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    polygon: np.ndarray
    color: Tuple[int,int,int]=(0,255,0)

def normalize_zone_config(zones: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out=[]
    for z in zones or []:
        pts=z.get("points") or z.get("polygon") or []
        out.append({
            "id": z.get("id", z.get("name","")),
            "name": z.get("name",""),
            "color": tuple(z.get("color",(0,255,0))),
            "points": [[float(x), float(y)] for x,y in pts],
        })
    return out

class ZoneRegistry:
    def __init__(self, config: List[Dict[str,Any]]):
        self.zones: List[Zone]=[]
        for z in normalize_zone_config(config):
            poly=np.array(z["points"], dtype=np.int32)
            self.zones.append(Zone(id=str(z["id"]), name=str(z["name"]), polygon=poly, color=tuple(z["color"])))
    def get_zone_for_point(self, x: float, y: float) -> Optional[Zone]:
        pt=(float(x), float(y))
        for zone in self.zones:
            if cv2.pointPolygonTest(zone.polygon, pt, False) >= 0:
                return zone
        return None

def anchor_point_xyxy(xyxy, mode="bottom_center"):
    x1,y1,x2,y2=map(float, xyxy)
    if mode=="center":
        return (0.5*(x1+x2), 0.5*(y1+y2))
    if mode=="bottom_left":
        return (x1, y2)
    if mode=="bottom_right":
        return (x2, y2)
    return (0.5*(x1+x2), y2)

def zone_for_bbox(xyxy, registry: ZoneRegistry, anchor_mode="bottom_center"):
    u,v=anchor_point_xyxy(xyxy, anchor_mode)
    z=registry.get_zone_for_point(u,v) if registry else None
    if z is None:
        return None, None, (u,v)
    return z.name, z.id, (u,v)
