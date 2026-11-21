"use client";
import Link from "next/link";
import { Activity, BarChart3, Camera, Database, Gauge, ShieldCheck, SlidersHorizontal } from "lucide-react";
import { usePathname } from "next/navigation";

const links = [{ href: "/", label: "Overview", icon: Gauge }, { href: "/detect", label: "Image detection", icon: Camera }, { href: "/history", label: "Detection history", icon: Database }, { href: "/analytics", label: "Analytics", icon: BarChart3 }, { href: "/model", label: "Model information", icon: SlidersHorizontal }];
export function Sidebar() { const pathname = usePathname(); return <aside className="sidebar"><div className="brand"><div className="brand-mark"><ShieldCheck size={21} /></div><div><strong>VISIONGUARD</strong><span>AI OPERATIONS</span></div></div><nav>{links.map(({ href, label, icon: Icon }) => <Link className={pathname === href ? "nav-link active" : "nav-link"} href={href} key={href}><Icon size={18} />{label}</Link>)}</nav><div className="sidebar-footer"><Activity size={16} /><span>Inference service</span><i /></div></aside> }
