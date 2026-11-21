"use client";
import { useEffect, useState } from "react";
import { BarChart3 } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/lib/api";
import type { Detection } from "@/types";

export default function Analytics() {
  const [items, setItems] = useState<Detection[]>([]);
  useEffect(() => { api.history().then((x) => setItems(x.items)).catch(() => undefined); }, []);
  const total = items.reduce((n, x) => n + x.detection_count, 0);
  const chartData = items.slice(0, 10).reverse().map((item) => ({ name: item.filename.length > 12 ? `${item.filename.slice(0, 12)}…` : item.filename, detections: item.detection_count }));
  return <><div className="topbar"><div><div className="kicker">Workspace / analytics</div><h1 className="title">Detection analytics</h1><p className="subtitle">Charts are derived only from records created in this deployment.</p></div></div>{items.length?<div className="grid-2"><section className="card"><div className="card-head"><span className="card-title">Detection volume</span><BarChart3 size={18} color="#57d6d0"/></div><div className="metric-value">{total}</div><p className="subtitle">Total detected persons across {items.length} images.</p><div style={{ height: 240, marginTop: 22 }}><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData}><CartesianGrid stroke="#202938" vertical={false}/><XAxis dataKey="name" stroke="#8792a6" tick={{ fontSize: 10 }}/><YAxis stroke="#8792a6" tick={{ fontSize: 10 }}/><Tooltip contentStyle={{ background: "#111722", border: "1px solid #202938" }}/><Bar dataKey="detections" fill="#57d6d0" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer></div></section><section className="card"><div className="card-head"><span className="card-title">Confidence distribution</span></div><div className="empty">Confidence aggregation will populate as detection records grow.</div></section></div>:<section className="card"><div className="empty">No live analytics yet.<br/>Run an image detection to generate workspace data.</div></section>}</>;
}
