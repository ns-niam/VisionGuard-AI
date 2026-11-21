import type { ReactNode } from "react";
export function MetricCard({ label, value, hint, icon }: { label: string; value: string; hint: string; icon: ReactNode }) { return <div className="metric-card"><div className="metric-icon">{icon}</div><div className="eyebrow">{label}</div><div className="metric-value">{value}</div><div className="metric-hint">{hint}</div></div> }
