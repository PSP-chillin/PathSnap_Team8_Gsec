"""Temporal / continuity features. Input: output of gps_features.add_gps_features."""
import numpy as np
import pandas as pd

from .geo import angular_diff_deg

GAP_S = 5.0          # dt above this is a GPS gap (97% of steps are exactly 1 s)
MAX_SPEED_MPS = 40.0  # ~144 km/h
MAX_ACCEL = 6.0      # m/s^2, beyond this speed change is implausible for a road vehicle
MAX_TURN_DEG = 90.0


def add_temporal_features(df: pd.DataFrame, gap_s=GAP_S, window=5, max_speed=MAX_SPEED_MPS,
                          max_turn_deg=MAX_TURN_DEG, max_accel=MAX_ACCEL) -> pd.DataFrame:
    out = df.copy()
    out["dt_next"] = out["time"].diff().shift(-1)
    out["is_gap_prev"] = (out["dt_prev"] > gap_s).astype(int)
    out["is_gap_next"] = (out["dt_next"] > gap_s).astype(int)
    out["dt_irregular"] = ((out["dt_prev"] - 1.0).abs() > 1e-6).astype(int)  # nominal rate is 1 Hz
    out["segment_id"] = out["is_gap_prev"].cumsum()           # contiguous (gap-free) sub-trajectory
    out["pos_in_segment"] = out.groupby("segment_id").cumcount()
    out["trip_progress"] = (out["time"] - out["time"].iloc[0]) / max(out["time"].iloc[-1] - out["time"].iloc[0], 1e-9)

    out["speed_change"] = out["speed"].diff()
    roll = out["speed"].rolling(window, center=True, min_periods=2)
    out["speed_roll_mean"], out["speed_roll_std"] = roll.mean(), roll.std()
    out["speed_cv"] = out["speed_roll_std"] / out["speed_roll_mean"].where(out["speed_roll_mean"] > 0.5)

    out["heading_change_abs"] = out["heading_change"].abs()
    out["heading_diff_prev_next"] = angular_diff_deg(out["heading_prev"], out["heading_next"])
    out["heading_roll_mean_turn"] = out["heading_change_abs"].rolling(window, center=True, min_periods=2).mean()

    ok = (
        (out["is_gap_prev"] == 0) & (out["is_gap_next"] == 0)
        & (out["speed"].fillna(0) <= max_speed)
        & (out["speed_change"].abs().fillna(0) <= max_accel * out["dt_prev"].clip(lower=1).fillna(1))
        & (out["heading_change_abs"].fillna(0) <= max_turn_deg)
    )
    out["is_continuous"] = ok.astype(int)
    return out


def trip_sampling_summary(df: pd.DataFrame, gap_s=GAP_S) -> dict:
    dt = df["dt_prev"].dropna()
    return {"n_obs": len(df), "dt_median": float(dt.median()), "dt_max": float(dt.max()),
            "frac_nominal_1s": float((dt.sub(1).abs() < 1e-6).mean()), "n_gaps": int((dt > gap_s).sum())}