// colormap.ts — divergent blue↔red colormap for the activation heatmap.
// Per-layer-normalized: t = clamp(v / layerMax, -1, 1); t≥0 → red, t<0 → blue,
// intensity = |t|; near-zero → dark neutral.
function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/** Map a value to an `rgb(r,g,b)` string, given the layer's max |activation|. */
export function divergentColor(value: number, layerMax: number): string {
  if (layerMax <= 0) return "rgb(24,28,38)"; // dark neutral when all-zero
  const t = clamp(value / layerMax, -1, 1);
  const mag = Math.abs(t);
  if (t >= 0) {
    // red, dimmer as magnitude shrinks
    const r = Math.round(40 + 200 * mag);
    const g = Math.round(40 + 20 * (1 - mag));
    const b = Math.round(50 + 20 * (1 - mag));
    return `rgb(${r},${g},${b})`;
  } else {
    const r = Math.round(40 + 20 * (1 - mag));
    const g = Math.round(40 + 20 * (1 - mag));
    const b = Math.round(60 + 195 * mag);
    return `rgb(${r},${g},${b})`;
  }
}

/** Max absolute value across a layer's activations (for normalization). */
export function maxAbs(values: ReadonlyArray<number>): number {
  let m = 0;
  for (const v of values) {
    const a = Math.abs(v);
    if (a > m) m = a;
  }
  return m;
}

// --- per-layer-type colormaps (req task 8) ---------------------------------
// These return CSS `color-mix()` / var() strings so the heatmap respects the
// active light/dark theme via the --x-color / --o-color / --hidden-color vars.

/** Round a 0..1 intensity to an integer mix percentage. */
function pct(intensity: number): number {
  return Math.round(clamp(intensity, 0, 1) * 100);
}

/**
 * Input row: color by X/O meaning. value > 0 → X color, value < 0 → O color,
 * value 0 (empty) → neutral elevated background.
 */
export function inputCellColor(value: number): string {
  if (value > 0) return "var(--x-color)";
  if (value < 0) return "var(--o-color)";
  return "var(--bg-elev)";
}

/**
 * Hidden row: yellow tint whose intensity is |value| / layerMax (ReLU acts are
 * ≥ 0, but abs keeps it robust). Mixes --hidden-color into the background.
 */
export function hiddenCellColor(value: number, layerMax: number): string {
  const intensity = layerMax > 0 ? Math.abs(value) / layerMax : 0;
  return `color-mix(in srgb, var(--hidden-color) ${pct(intensity)}%, var(--bg-elev))`;
}

/**
 * Output row: tint by the AI's side color (X for side +1, O for side −1) with
 * intensity = prob / maxProb, so the favored move is brightest.
 */
export function outputCellColor(prob: number, maxProb: number, side: 1 | -1): string {
  const intensity = maxProb > 0 ? prob / maxProb : 0;
  const sideVar = side > 0 ? "var(--x-color)" : "var(--o-color)";
  return `color-mix(in srgb, ${sideVar} ${pct(intensity)}%, var(--bg-elev))`;
}

/** Numerically stable softmax over the raw output Q-values. */
export function softmax(values: number[]): number[] {
  if (values.length === 0) return [];
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return sum > 0 ? exps.map((e) => e / sum) : exps.map(() => 0);
}
