// Copy /data exports into public/data for the static site (goldens and splits are test-only).
import fs from "node:fs";
import path from "node:path";

const src = path.resolve(import.meta.dirname, "../../data");
const dst = path.resolve(import.meta.dirname, "../public/data");
const skip = new Set(["golden", "splits.json"]);
fs.rmSync(dst, { recursive: true, force: true });
fs.cpSync(src, dst, { recursive: true, filter: (p) => !skip.has(path.basename(p)) || path.dirname(p) !== src });
console.log(`copied ${src} -> ${dst}`);
