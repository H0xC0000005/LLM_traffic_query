from collections import defaultdict
import math
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Tuple, Protocol


# NEW: pluggable preprocessor contract
class ScenePreprocessor(Protocol):
    def run(self, scene: dict) -> Dict[str, Any]:
        """Return a dict of {symbol_name: value}. Values can be str or Python objects."""
        ...

class BasicStatsPreprocessor:
    """
    Basic scene preprocessor that computes only a few simple statistics:
        - vehicle_count: total number of vehicles
        - lane_count: number of unique lanes occupied
        - speed_kmh_min/max/avg: global min/max/avg of vehicle speeds
    """
    def run(self, scene: dict) -> Dict[str, Any]:
        vehicles = (scene or {}).get("vehicles", []) or []
        lanes = set()
        speeds = []
        for v in vehicles:
            lane = v.get("lane")
            if lane is not None:
                lanes.add(lane)
            sp = v.get("speed_kmh")
            if isinstance(sp, (int, float)):
                speeds.append(float(sp))
        stats = {
            "vehicle_count": len(vehicles),
            "lane_count": len(lanes),
            "speed_kmh_min": min(speeds) if speeds else None,
            "speed_kmh_max": max(speeds) if speeds else None,
            "speed_kmh_avg": (sum(speeds) / len(speeds)) if speeds else None,
        }
        # Expose under a conventional symbol name
        return {"scene_stats": stats}
    
class FullStatsPreprocessor:
    """
    Computes richer scene statistics:
      - per_class_speed: min/max/avg/std for each vehicle type
      - per_lane_speed:  min/max/avg/std for each lane
      - per_lane_interveh_dist_m: mean/std/min/max of center-to-center distances (by lane)
      - per_lane_type_counts: counts of each vehicle type per lane
      - retains: vehicle_count, lane_count (supersedes old global speed stats)
    Notes:
      * STD is 'sample' (ddof=1). Returns None if fewer than 2 values.
      * Distances are computed along the longitudinal axis using vehicle.coord[1].
    """
    def __init__(self, round_ndigits: Optional[int] = 4, std_mode: str = "sample"):
        self.round_ndigits = round_ndigits
        self.ddof = 1 if std_mode == "sample" else 0  # 'sample' or 'population'

    # ---------- numeric helpers ----------
    def _mean(self, xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None
    
    def _median(self, xs: List[float]) -> Optional[float]:
        n = len(xs)
        if n == 0:
            return None
        xs_sorted = sorted(xs)
        mid = n // 2
        if n % 2 == 1:
            return xs_sorted[mid]
        else:
            return (xs_sorted[mid - 1] + xs_sorted[mid]) / 2.0

    def _std(self, xs: List[float]) -> Optional[float]:
        n = len(xs)
        if n == 0:
            return None
        if n == 1:
            return None if self.ddof == 1 else 0.0
        mu = self._mean(xs)
        var = sum((x - mu) ** 2 for x in xs) / (n - self.ddof)
        return math.sqrt(var)

    def _agg_stats(self, xs: List[float]) -> Dict[str, Optional[float]]:
        if not xs:
            return {"min": None, "max": None, "avg": None, "std": None, "median": None}
        out = {
            "min": min(xs),
            "max": max(xs),
            "avg": self._mean(xs),
            "std": self._std(xs),
            "med": self._median(xs),
        }
        if self.round_ndigits is not None:
            for k, v in out.items():
                if isinstance(v, float):
                    out[k] = round(v, self.round_ndigits)
        return out

    # ---------- extraction helpers ----------
    def _get_speed(self, v: Dict[str, Any]) -> Optional[float]:
        sp = v.get("speed_kmh")
        if isinstance(sp, (int, float)):
            return float(sp)
        return None

    def _get_lane(self, v: Dict[str, Any]) -> Optional[int]:
        ln = v.get("lane")
        if isinstance(ln, int):
            return ln
        # tolerate numeric-as-str
        if isinstance(ln, str) and ln.isdigit():
            return int(ln)
        return None

    def _get_longitudinal(self, v: Dict[str, Any]) -> Optional[float]:
        """
        Extract longitudinal coordinate. Preferred: v["coord"][1].
        Fallbacks can be added here if your schema evolves (e.g., v["s"]).
        """
        coord = v.get("coord")
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            try:
                return float(coord[1])
            except Exception:
                return None
        # Example fallback: if you later add 's' (station)
        s = v.get("s")
        if isinstance(s, (int, float)):
            return float(s)
        return None

    # ---------- main entry ----------
    def run(self, scene: dict) -> Dict[str, Any]:
        vehicles: List[Dict[str, Any]] = (scene or {}).get("vehicles", []) or []

        # buckets
        lane_ids = set()
        speeds_by_type: DefaultDict[str, List[float]] = defaultdict(list)
        speeds_by_lane: DefaultDict[int, List[float]] = defaultdict(list)
        coords_by_lane: DefaultDict[int, List[float]] = defaultdict(list)
        counts_by_lane_type: DefaultDict[int, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

        # collect
        for v in vehicles:
            vtype = str(v.get("type", "unknown"))
            lane = self._get_lane(v)
            sp = self._get_speed(v)
            y = self._get_longitudinal(v)

            if lane is not None:
                lane_ids.add(lane)
                counts_by_lane_type[lane][vtype] += 1
                if sp is not None:
                    speeds_by_lane[lane].append(sp)
                if y is not None:
                    coords_by_lane[lane].append(y)

            if sp is not None:
                speeds_by_type[vtype].append(sp)

        # per-class speed stats
        per_class_speed: Dict[str, Any] = {}
        for vtype, xs in sorted(speeds_by_type.items()):
            s = self._agg_stats(xs)
            per_class_speed[vtype] = {"count": len(xs),
                                      "speed_kmh_min": s["min"],
                                      "speed_kmh_max": s["max"],
                                      "speed_kmh_avg": s["avg"],
                                      "speed_kmh_std": s["std"],
                                      "speed_kmh_med": s["med"]}

        # per-lane speed stats
        per_lane_speed: Dict[str | int, Any] = {}
        for lane, xs in sorted(speeds_by_lane.items()):
            s = self._agg_stats(xs)
            per_lane_speed[lane] = {"count": len(xs),
                                    "speed_kmh_min": s["min"],
                                    "speed_kmh_max": s["max"],
                                    "speed_kmh_avg": s["avg"],
                                    "speed_kmh_std": s["std"],
                                    "speed_kmh_med": s["med"]}

        # per-lane inter-vehicle distances
        per_lane_interveh_dist_m: Dict[str | int, Any] = {}
        for lane, ys in sorted(coords_by_lane.items()):
            if len(ys) < 2:
                per_lane_interveh_dist_m[lane] = {
                    "pair_count": 0,
                    "min": None, "max": None, "mean": None, "std": None,
                }
                continue
            ys_sorted = sorted(ys)
            gaps = [ys_sorted[i+1] - ys_sorted[i] for i in range(len(ys_sorted) - 1)]
            # Make distances positive; if your coordinate always increases forward, gaps should already be >= 0
            gaps = [abs(g) for g in gaps]

            stats = self._agg_stats(gaps)
            # rename keys for this block
            out = {
                "pair_count": len(gaps),
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["avg"],
                "std": stats["std"],
            }
            per_lane_interveh_dist_m[lane] = out

        # per-lane per-type counts (convert default dicts to plain dicts)
        per_lane_type_counts: Dict[str | int, Dict[str, int]] = {}
        for lane, typemap in sorted(counts_by_lane_type.items()):
            per_lane_type_counts[lane] = dict(sorted(typemap.items()))

        # retain original fields we didn't supersede
        scene_stats: Dict[str, Any] = {
            "vehicle_count": len(vehicles),
            "lane_count": len(lane_ids),

            # new blocks
            "per_class_speed": per_class_speed,
            "per_lane_speed": per_lane_speed,
            "per_lane_interveh_dist_m": per_lane_interveh_dist_m,
            "per_lane_type_counts": per_lane_type_counts,

            # meta to aid consumers
            "_notes": {
                "speeds_unit": "km/h",
                "distance_unit": "m",
                "std_mode": "sample" if self.ddof == 1 else "population",
                "std_is_none_if_n<2": True,
            },
        }

        return {"scene_stats": scene_stats}