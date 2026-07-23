// chart-svg.ts — hand-rolled SVG line chart, no chart library. Holds a ring
// buffer of values and redraws a single <polyline> with an auto-scaled Y axis
// on demand. Clutter-free: one baseline + a current-value label.
export class RingBuffer<T> {
  private data: T[] = [];
  constructor(public capacity: number) {}

  push(v: T): void {
    this.data.push(v);
    if (this.data.length > this.capacity) this.data.shift();
  }

  get values(): readonly T[] {
    return this.data;
  }

  clear(): void {
    this.data = [];
  }
}

interface ChartOpts {
  minY: number;
  maxY: number; // fixed max; pass dynamic value to rescale
  formatValue?: (v: number) => string;
}

/** Redraw an SVG element from a numeric ring buffer. */
export function drawChart(
  svg: SVGSVGElement,
  buffer: RingBuffer<number>,
  opts: ChartOpts,
): void {
  const W = 200;
  const H = 60;
  const vals = buffer.values;
  // Clear previous content (keep the baseline we add below).
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const ns = "http://www.w3.org/2000/svg";

  // baseline at the bottom
  const baseline = document.createElementNS(ns, "line");
  baseline.setAttribute("x1", "0");
  baseline.setAttribute("y1", String(H - 0.5));
  baseline.setAttribute("x2", String(W));
  baseline.setAttribute("y2", String(H - 0.5));
  baseline.setAttribute("class", "chart-baseline");
  svg.appendChild(baseline);

  if (vals.length < 2) return;

  const { minY, maxY } = opts;
  const range = Math.max(1e-9, maxY - minY);
  const n = vals.length;
  const pts: string[] = [];
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * W;
    const y = H - ((vals[i]! - minY) / range) * H;
    pts.push(`${x.toFixed(2)},${y.toFixed(2)}`);
  }
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", pts.join(" "));
  line.setAttribute("fill", "none");
  line.setAttribute("class", "chart-line");
  svg.appendChild(line);

  // current-value label
  const last = vals[n - 1]!;
  const label = document.createElementNS(ns, "text");
  label.setAttribute("x", String(W - 2));
  label.setAttribute("y", "10");
  label.setAttribute("text-anchor", "end");
  label.setAttribute("class", "chart-label");
  label.textContent = opts.formatValue ? opts.formatValue(last) : last.toFixed(3);
  svg.appendChild(label);
}
