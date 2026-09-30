import type { NextConfig } from "next";

// Static export: the whole guide is plain files (HTML/JS + /data), no server.
// NEXT_PUBLIC_BASE_PATH serves it from a subpath (e.g. "/jepa-visualization-" on GitHub Pages).
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || undefined;

const config: NextConfig = {
  output: "export",
  basePath,
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default config;
