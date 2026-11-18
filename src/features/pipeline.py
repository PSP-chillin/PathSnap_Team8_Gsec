"""Convenience: cleaned trip dict -> all feature tables for one trip."""
from .candidate_features import Geometry, candidate_features, generate_candidates
from .gps_features import add_gps_features
from .network_features import RoadNetwork, candidate_local_features, local_arc_properties, transition_features
from .temporal_features import add_temporal_features


def build_trip_features(trip, k=10, radius_m=100.0):
    gps = add_temporal_features(add_gps_features(trip["track"]))
    geom = Geometry(trip["nodes"], trip["arcs"])
    px, py = geom.project_track(gps)
    cands = generate_candidates(px, py, geom, k=k, radius_m=radius_m)
    cf = candidate_features(gps, px, py, cands, geom)
    net = RoadNetwork(geom.arc_table, len(trip["nodes"]))
    cf = candidate_local_features(cf, local_arc_properties(geom.arc_table, len(trip["nodes"])))
    tr = transition_features(cf, gps, px, py, net)
    return gps, cf, tr, geom, px, py