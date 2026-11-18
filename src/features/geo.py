"""Shared geometry helpers (metres, degrees clockwise from north)."""
import numpy as np

EARTH_R = 6_371_000.0


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(lambda a: np.radians(np.asarray(a, float)), (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def initial_bearing_deg(lon1, lat1, lon2, lat2):
    """Great-circle initial bearing in [0, 360), 0 = north, 90 = east."""
    lon1, lat1, lon2, lat2 = map(lambda a: np.radians(np.asarray(a, float)), (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def make_projector(ref_lon, ref_lat):
    """Local equirectangular projection (metres) around a reference point.
    Accurate to well under 0.1% at city scale, which is all PathSnap needs."""
    k = np.cos(np.radians(ref_lat))

    def project(lon, lat):
        x = np.radians(np.asarray(lon, float) - ref_lon) * EARTH_R * k
        y = np.radians(np.asarray(lat, float) - ref_lat) * EARTH_R
        return x, y

    return project


def xy_bearing_deg(dx, dy):
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angular_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]. NaN-safe."""
    d = np.abs((np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0)
    return d


def signed_angular_diff_deg(a, b):
    """(a - b) wrapped to [-180, 180)."""
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0