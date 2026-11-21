import type { Config } from "tailwindcss";
export default { content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"], theme: { extend: { colors: { ink: "#080b12", panel: "#111722", line: "#202938", cyan: "#57d6d0", amber: "#f6b654" } } }, plugins: [] } satisfies Config;
