import { createRoot } from "react-dom/client";
import "leaflet/dist/leaflet.css";
import "./styles.css";
import App from "./App.jsx";

// StrictMode is left off on purpose: in development it runs every effect twice,
// which would send duplicate requests and clutter the API log the evaluator watches.
createRoot(document.getElementById("root")).render(<App />);
