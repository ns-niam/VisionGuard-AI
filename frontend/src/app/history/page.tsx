"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowUpRight, Search } from "lucide-react";
import { api } from "@/lib/api";
import type { Detection } from "@/types";

export default function History() {
  const [items, setItems] = useState<Detection[]>([]);
  const [search, setSearch] = useState("");
  useEffect(() => { api.history(search).then((x) => setItems(x.items)).catch(() => undefined); }, [search]);
  return <><div className="topbar"><div><div className="kicker">Workspace / records</div><h1 className="title">Detection history</h1><p className="subtitle">A private record of analyses run through this workspace.</p></div></div><section className="card"><div className="card-head"><span className="card-title">Saved analyses</span><span className="card-note">{items.length} records</span></div><div style={{ position: "relative", marginBottom: 16 }}><Search size={15} style={{ position: "absolute", left: 11, top: 11, color: "#8792a6" }} /><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search filenames" style={{ width: "100%", background: "#0c121b", border: "1px solid #202938", borderRadius: 6, color: "#edf2f7", padding: "10px 12px 10px 34px" }} /></div>{items.length?<table className="table"><thead><tr><th>Filename</th><th>Detections</th><th>Dimensions</th><th>Latency</th><th></th></tr></thead><tbody>{items.map((x)=><tr key={x.id}><td>{x.filename}</td><td className="accent">{x.detection_count}</td><td>{x.image_width} × {x.image_height}</td><td>{x.inference_latency_ms.toFixed(0)} ms</td><td><Link href={`/history/${x.id}`} className="accent"><ArrowUpRight size={16}/></Link></td></tr>)}</tbody></table>:<div className="empty">{search ? "No matching records." : "No saved analyses yet."}</div>}</section></>;
}
