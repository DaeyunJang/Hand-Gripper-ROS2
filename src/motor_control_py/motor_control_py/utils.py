from __future__ import annotations

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def rad_to_tick_linear(rad: float,
                       rmin: float, rmax: float,
                       tmin: int, tmax: int) -> int:
    """라디안 [rmin,rmax] → 틱 [tmin,tmax] 선형 변환(+클램프)."""
    if rmax <= rmin:
        raise ValueError("rmax must be > rmin")
    x = (rad - rmin) / (rmax - rmin)
    x = clamp(x, 0.0, 1.0)
    return int(round(tmin + x * (tmax - tmin)))
