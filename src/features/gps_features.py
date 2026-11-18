"""Per-observation GPS features. Input: cleaned `track` DataFrame (longitude, latitude, time)."""
import numpy as np
import pandas as pd

from .geo import haversine_m, initial_bearing_deg, signed_angular_diff_deg

MIN_MOVE_M = 1.0  # below this, heading is considered unreliable (GPS jitter)


def add_gps_features(track: pd.DataFrame, min_move_m: float = MIN_MOVE_M) -> pd.DataFrame:
    """Return a copy of `track` with GPS motion features (row order preserved, index = obs_idx).

    Backward quantities use the previous point (NaN on the first row); forward quantities
    use the next point (NaN on the last row).
    """
    df = track.reset_index(drop=True).copy()
    lon, lat, t = df["longitude"].to_numpy(float), df["latitude"].to_numpy(float), df["time"].to_numpy(float)

    d_prev = np.full(len(df), np.nan)
    d_prev[1:] = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    df["dist_prev"] = d_prev
    df["dist_next"] = df["dist_prev"].shift(-1)

    dt = df["time"].diff()
    df["dt_prev"] = dt
    df["speed"] = np.where(dt > 0, df["dist_prev"] / dt.where(dt > 0), np.nan)  # m/s, NaN if dt<=0
    df["speed_next"] = df["speed"].shift(-1)

    brg = np.full(len(df), np.nan)
    brg[1:] = initial_bearing_deg(lon[:-1], lat[:-1], lon[1:], lat[1:])
    brg = np.where(d_prev >= min_move_m, brg, np.nan)
    df["heading_prev"] = brg                      # bearing prev -> current
    df["heading_next"] = pd.Series(brg).shift(-1).to_numpy()  # bearing current -> next
    # Main heading: backward bearing, falling back to forward bearing (first point / stationary)
    df["heading"] = df["heading_prev"].fillna(df["heading_next"])
    df["is_stationary"] = (df["speed"].fillna(0) < 0.5).astype(int)

    df["heading_change"] = signed_angular_diff_deg(df["heading_next"], df["heading_prev"])  # turn at this point
    df["acceleration"] = (df["speed_next"] - df["speed"]) / df["time"].diff().shift(-1).where(lambda s: s > 0)
    # displacement over a 5-point window (net movement; small vs path length => zig-zag/noise)
    w = 4
    disp = np.full(len(df), np.nan)
    if len(df) > w:
        disp[w:] = haversine_m(lon[:-w], lat[:-w], lon[w:], lat[w:])
    path = df["dist_prev"].rolling(w, min_periods=w).sum().to_numpy()
    df["displacement_w"] = disp
    df["path_length_w"] = path
    df["straightness_w"] = np.where(path > 0, disp / path, np.nan)
    return df