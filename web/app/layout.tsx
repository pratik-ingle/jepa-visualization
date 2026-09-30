import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "JEPA Visualization",
  description: "An interactive 3D guide to I-JEPA and LeJEPA: two tiny joint-embedding predictive architectures running live in the browser.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#0b0e13" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning: browser extensions (e.g. Scholarcy, Grammarly, Dark Reader) inject
    // attributes into <html>/<body> before React hydrates. It only applies to these two elements'
    // own attributes, so mismatches inside the app are still reported.
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased" suppressHydrationWarning>{children}</body>
    </html>
  );
}
