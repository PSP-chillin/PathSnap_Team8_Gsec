import { useEffect, useMemo } from "react";
import L from "leaflet";
import { CircleMarker, LayerGroup, LayersControl, MapContainer, Pane, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";

// Drawing tens of thousands of markers slows the page; above this, every Nth point is drawn.
const MAX_GPS_MARKERS = 3000;

const COLORS = {
  gps: "#2563EB",
  route: "#E8590C",
  start: "#16A34A",
  end: "#DC2626",
};

function FitToResult({ result }) {
  const map = useMap();
  useEffect(() => {
    if (!result) return;
    const points = [...result.gpsPoints, ...result.routeSegments.flat()];
    if (points.length === 0) return;
    map.fitBounds(L.latLngBounds(points), { padding: [32, 32], maxZoom: 17 });
  }, [map, result]);
  return null;
}

function Legend({ result, step }) {
  return (
    <div className="map-legend" aria-label="Map legend">
      <div className="legend-row"><span className="swatch swatch-dot" style={{ background: COLORS.gps }} />GPS points (input)</div>
      <div className="legend-row"><span className="swatch swatch-line" style={{ background: COLORS.route }} />Predicted route</div>
      <div className="legend-row"><span className="swatch swatch-ring" style={{ borderColor: COLORS.start }} />Start</div>
      <div className="legend-row"><span className="swatch swatch-ring" style={{ borderColor: COLORS.end }} />End</div>
      {result && step > 1 && (
        <div className="legend-note">
          Drawing 1 in every {step} of {result.gpsPoints.length.toLocaleString()} GPS points
        </div>
      )}
    </div>
  );
}

export default function MapView({ result, runningId, emptyMessage = "Select a trip to see its predicted route" }) {
  const { points, step } = useMemo(() => {
    if (!result) return { points: [], step: 1 };
    const every = Math.max(1, Math.ceil(result.gpsPoints.length / MAX_GPS_MARKERS));
    return {
      points: every === 1 ? result.gpsPoints : result.gpsPoints.filter((_, i) => i % every === 0),
      step: every,
    };
  }, [result]);

  const start = result?.gpsPoints[0];
  const end = result?.gpsPoints[result.gpsPoints.length - 1];

  return (
    <div className="map-shell">
      <MapContainer center={[20, 0]} zoom={2} minZoom={2} preferCanvas className="map" worldCopyJump>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          maxZoom={19}
        />
        <LayersControl position="topright">
          <LayersControl.Overlay checked name="GPS points (input)">
            <LayerGroup>
              {points.map((point, i) => (
                <CircleMarker
                  key={i}
                  center={point}
                  radius={2.5}
                  pathOptions={{ color: COLORS.gps, weight: 1, fillColor: COLORS.gps, fillOpacity: 0.7 }}
                />
              ))}
            </LayerGroup>
          </LayersControl.Overlay>
          <LayersControl.Overlay checked name="Predicted route">
            {/* Own pane above the GPS dots; with one shared canvas the dots would cover the line. */}
            <Pane name="predicted-route" style={{ zIndex: 450 }}>
              <Polyline
                positions={result?.routeSegments ?? []}
                pathOptions={{ color: COLORS.route, weight: 5, opacity: 0.85, lineCap: "round", lineJoin: "round" }}
              />
            </Pane>
          </LayersControl.Overlay>
        </LayersControl>
        <Pane name="trip-ends" style={{ zIndex: 460 }}>
          {start && (
            <CircleMarker center={start} radius={7} pathOptions={{ color: COLORS.start, weight: 3, fillColor: "#fff", fillOpacity: 1 }}>
              <Tooltip direction="top">Start</Tooltip>
            </CircleMarker>
          )}
          {end && (
            <CircleMarker center={end} radius={7} pathOptions={{ color: COLORS.end, weight: 3, fillColor: "#fff", fillOpacity: 1 }}>
              <Tooltip direction="top">End</Tooltip>
            </CircleMarker>
          )}
        </Pane>
        <FitToResult result={result} />
      </MapContainer>

      <Legend result={result} step={step} />

      {!result && !runningId && <div className="map-empty">{emptyMessage}</div>}
      {runningId && (
        <div className="map-busy" role="status">
          <span className="spinner" /> Predicting trip {runningId}…
        </div>
      )}
    </div>
  );
}
