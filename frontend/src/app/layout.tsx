import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";
export const metadata: Metadata = { title: "VisionGuard AI", description: "Production pedestrian detection operations" };
export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <Sidebar />
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
