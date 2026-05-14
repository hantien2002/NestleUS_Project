import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from .io import load_json

@dataclass
class TriggerConfig:
    camera_id: str = "cam_demo_01"
    track_ttl_s: float = 1.0
    min_track_age_s: float = 0.3
    window_size_frames: int = 60
    violation_threshold_frames: int = 30
    reminder_interval_s: float = 3.0
    cooldown_s: float = 1.0

@dataclass
class AlertConfig:
    severity_weights: Dict[str, int] = field(default_factory=lambda: {
        "NO_HELMET_AND_VEST": 3,
        "NO_HELMET": 2,
        "NO_VEST": 1
    })

DEFAULT_ZONE_RULES = {
    "zone_a": {"helmet": True, "vest": True},
}

@dataclass
class TrackState:
    track_id: int
    first_seen_t: float
    last_seen_t: float
    zone_id: Optional[str] = None
    helmet_history: List[bool] = field(default_factory=list)
    vest_history: List[bool] = field(default_factory=list)
    is_violation_active: bool = False
    active_violation_type: Optional[str] = None
    violation_start_t: Optional[float] = None
    last_reminder_t: Optional[float] = None
    cooldown_until_t: float = -1.0

class ViolationTracker:
    def __init__(self, config: TriggerConfig, zone_rules: Optional[Dict[str,Dict[str,bool]]]=None):
        self.cfg=config
        self.zone_rules=zone_rules or DEFAULT_ZONE_RULES
        self.tracks: Dict[int,TrackState]={}
        self.events: List[dict]=[]
        self.current_sec=0
        self.second_logs: List[dict]=[]

    def _detect_violation_type(self, zone_id: str, has_helmet: bool, has_vest: bool) -> Optional[str]:
        if not zone_id or zone_id not in self.zone_rules: return None
        req=self.zone_rules[zone_id]
        miss_helmet = req.get("helmet", False) and not has_helmet
        miss_vest   = req.get("vest", False) and not has_vest
        if miss_helmet and miss_vest: return "NO_HELMET_AND_VEST"
        if miss_helmet: return "NO_HELMET"
        if miss_vest: return "NO_VEST"
        return None

    def _emit_event(self, t: float, track: TrackState, event_type: str, violation_type: str=None, extra: dict=None):
        payload={"time_sec": round(float(t),3), "track_id": track.track_id, "zone_id": track.zone_id,
                 "event_type": event_type, "violation_type": violation_type or track.active_violation_type}
        if extra: payload.update(extra)
        self.events.append(payload)

    def process_frame(self, time_sec: float, persons: List[dict]):
        time_sec=float(time_sec)
        current_track_ids=set()

        for p in persons or []:
            tid=int(p["id"])
            current_track_ids.add(tid)
            zone_id = p.get("zone_id") or p.get("zone_name")
            raw_has_helmet=bool(p.get("has_helmet", False))
            raw_has_vest=bool(p.get("has_vest", False))

            if tid not in self.tracks:
                self.tracks[tid]=TrackState(track_id=tid, first_seen_t=time_sec, last_seen_t=time_sec, zone_id=zone_id)
            st=self.tracks[tid]
            st.last_seen_t=time_sec

            if st.zone_id != zone_id:
                if st.is_violation_active:
                    self._emit_event(time_sec, st, "VIOLATION_RESOLVED", extra={"reason":"zone_change"})
                st.zone_id=zone_id
                st.is_violation_active=False
                st.active_violation_type=None
                st.violation_start_t=None
                st.helmet_history.clear()
                st.vest_history.clear()

            if (time_sec - st.first_seen_t) < self.cfg.min_track_age_s:
                continue

            st.helmet_history.append(raw_has_helmet)
            st.vest_history.append(raw_has_vest)
            if len(st.helmet_history) > self.cfg.window_size_frames:
                st.helmet_history.pop(0); st.vest_history.pop(0)

            if len(st.helmet_history) < self.cfg.window_size_frames:
                continue

            missing_helmet_count=st.helmet_history.count(False)
            missing_vest_count=st.vest_history.count(False)
            smoothed_has_helmet = (missing_helmet_count < self.cfg.violation_threshold_frames)
            smoothed_has_vest   = (missing_vest_count < self.cfg.violation_threshold_frames)
            current_violation=self._detect_violation_type(zone_id, smoothed_has_helmet, smoothed_has_vest)

            if not st.is_violation_active:
                if current_violation:
                    st.is_violation_active=True
                    st.active_violation_type=current_violation
                    st.violation_start_t=time_sec
                    st.cooldown_until_t=time_sec + self.cfg.cooldown_s
                    self._emit_event(time_sec, st, "VIOLATION_STARTED", current_violation, {"since": st.violation_start_t})
            else:
                if not current_violation:
                    self._emit_event(time_sec, st, "VIOLATION_RESOLVED", extra={"duration": time_sec - (st.violation_start_t or time_sec)})
                    st.is_violation_active=False
                    st.active_violation_type=None
                    st.violation_start_t=None
                elif current_violation != st.active_violation_type:
                    st.active_violation_type=current_violation

        self._cleanup_dead_tracks(time_sec, current_track_ids)

        current_int_sec=int(time_sec)
        if current_int_sec > self.current_sec:
            self.current_sec=current_int_sec
            for st in self.tracks.values():
                if not st.is_violation_active: 
                    continue
                m_helmet=st.helmet_history.count(False)
                m_vest=st.vest_history.count(False)
                w_size=len(st.helmet_history)
                if st.active_violation_type == "NO_HELMET_AND_VEST":
                    trigger_ratio=f"H_miss:{m_helmet}/{w_size} | V_miss:{m_vest}/{w_size}"
                elif st.active_violation_type == "NO_HELMET":
                    trigger_ratio=f"{m_helmet}/{w_size}"
                else:
                    trigger_ratio=f"{m_vest}/{w_size}"
                self.second_logs.append({
                    "second": current_int_sec,
                    "track_id": st.track_id,
                    "zone_id": st.zone_id,
                    "violation_type": st.active_violation_type,
                    "trigger_ratio": trigger_ratio,
                    "duration": round(time_sec - (st.violation_start_t or time_sec), 3)
                })

    def _cleanup_dead_tracks(self, current_time: float, present_ids: set):
        dead=[]
        for tid, st in list(self.tracks.items()):
            if tid in present_ids: 
                continue
            if (current_time - st.last_seen_t) > self.cfg.track_ttl_s:
                if st.is_violation_active:
                    self._emit_event(current_time, st, "EXIT_CAMERA", extra={"reason":"track_lost"})
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

def run_pipeline_per_second(json_input_path: str, output_dir: str, trigger_config: Optional[TriggerConfig]=None,
                            zone_rules: Optional[Dict[str,Dict[str,bool]]]=None,
                            alert_config: Optional[AlertConfig]=None):
    data=load_json(json_input_path)
    frames=data.get("frames", [])
    fps=float(data.get("fps", 30.0))
    trig_cfg=trigger_config or TriggerConfig()
    tracker=ViolationTracker(trig_cfg, zone_rules=zone_rules)

    for i, frame in enumerate(frames):
        t=float(frame.get("time_sec", i / fps))
        tracker.process_frame(t, frame.get("persons", []))

    final_time=float(frames[-1]["time_sec"]) if frames else 0.0
    tracker._cleanup_dead_tracks(final_time + trig_cfg.track_ttl_s + 1.0, set())

    out_dir=Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if not tracker.second_logs:
        return None

    alerts_df=pd.DataFrame(tracker.second_logs)
    sev=(alert_config or AlertConfig()).severity_weights
    alerts_df["severity"]=alerts_df["violation_type"].map(sev).fillna(1).astype(int)
    alerts_df=alerts_df[["second","track_id","zone_id","violation_type","trigger_ratio","duration","severity"]]
    csv_path=out_dir/"alerts_per_second.csv"
    alerts_df.to_csv(csv_path, index=False)
    return str(csv_path)
