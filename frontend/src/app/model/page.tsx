"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ModelInfo } from "@/types";

const testMetrics = [["Precision", 0.8003], ["Recall", 0.4319], ["F1", 0.561], ["mAP50", 0.5039], ["mAP50-95", 0.3103]] as const;
const comparison = [["Precision", 0.8394, 0.8003], ["Recall", 0.491, 0.4319], ["F1", 0.6196, 0.561], ["mAP50", 0.5931, 0.5039], ["mAP50-95", 0.3017, 0.3103]] as const;

export default function ModelPage() {
  const [model, setModel] = useState<ModelInfo>();
  useEffect(() => { api.model().then(setModel).catch(() => undefined); }, []);
  return <>
    <div className="topbar"><div><div className="kicker">Research artifact / model card</div><h1 className="title">Model information</h1><p className="subtitle">Transparent provenance for the production checkpoint.</p></div></div>
    <div className="grid-2">
      <section className="card"><div className="card-head"><span className="card-title">Selected checkpoint</span><span className="status-pill">VERIFIED</span></div><div className="info-grid"><div className="info-item"><small>Architecture</small><strong>{model?.architecture ?? "YOLO11s"}</strong></div><div className="info-item"><small>Parameters</small><strong>9,428,179</strong></div><div className="info-item"><small>Class 0</small><strong>person</strong></div><div className="info-item"><small>Inference image size</small><strong>640 px</strong></div></div><p className="subtitle" style={{ marginTop: 20 }}>Final model is selected using validation results only. It is loaded from <span className="accent">models/final/best.pt</span>.</p></section>
      <section className="card"><div className="card-head"><span className="card-title">Held-out test metrics</span><span className="card-note">Never used for selection</span></div><div className="metric-bars">{testMetrics.map(([label, value]) => <div key={label}><div className="bar-label"><span>{label}</span><span className="accent">{value.toFixed(4)}</span></div><div className="bar"><i style={{ width: `${value * 100}%` }} /></div></div>)}</div></section>
    </div>
    <section className="card" style={{ marginTop: 16 }}><div className="card-head"><span className="card-title">Validation vs held-out test</span></div><table className="table"><thead><tr><th>Metric</th><th>Validation</th><th>Held-out test</th></tr></thead><tbody>{comparison.map(([label, validation, test]) => <tr key={label}><td>{label}</td><td>{validation.toFixed(4)}</td><td className="accent">{test.toFixed(4)}</td></tr>)}</tbody></table></section>
  </>;
}
