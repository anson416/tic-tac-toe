// optimizer.ts — Adam optimizer state + step, plus weight initialization.
//
// Adam is chosen over vanilla SGD for a tiny net trained in-browser with
// limited wall-clock: per-parameter adaptive step sizes converge faster and
// are far less LR-sensitive on the sparse tic-tac-toe signal; cost is
// negligible (a few hundred params).

export interface AdamParam {
  m: Float64Array; // first moment
  v: Float64Array; // second moment
}

export interface AdamState {
  beta1: number;
  beta2: number;
  eps: number;
  t: number; // global step counter
  w: AdamParam; // for weights (same shape as the weight matrix)
  b: AdamParam; // for biases
}

export function makeAdam(wLen: number, bLen: number): AdamState {
  return {
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    t: 0,
    w: { m: new Float64Array(wLen), v: new Float64Array(wLen) },
    b: { m: new Float64Array(bLen), v: new Float64Array(bLen) },
  };
}

/**
 * In-place Adam update on a single weight matrix `W` (row-major, shape out×in)
 * and bias `b`, given averaged gradients `gradW`/`gradB`. Mutates W, b, and the
 * Adam state.
 */
export function adamStep(
  state: AdamState,
  W: Float64Array,
  b: Float64Array,
  gradW: Float64Array,
  gradB: Float64Array,
  lr: number,
): void {
  state.t += 1;
  const { beta1, beta2, eps, t } = state;
  const bc1 = 1 - Math.pow(beta1, t);
  const bc2 = 1 - Math.pow(beta2, t);
  const { m: mw, v: vw } = state.w;
  const { m: mb, v: vb } = state.b;
  for (let i = 0; i < W.length; i++) {
    const g = gradW[i];
    mw[i] = beta1 * mw[i] + (1 - beta1) * g;
    vw[i] = beta2 * vw[i] + (1 - beta2) * g * g;
    W[i] -= (lr * (mw[i] / bc1)) / (Math.sqrt(vw[i] / bc2) + eps);
  }
  for (let i = 0; i < b.length; i++) {
    const g = gradB[i];
    mb[i] = beta1 * mb[i] + (1 - beta1) * g;
    vb[i] = beta2 * vb[i] + (1 - beta2) * g * g;
    b[i] -= (lr * (mb[i] / bc1)) / (Math.sqrt(vb[i] / bc2) + eps);
  }
}

// --- Weight initialization -------------------------------------------------

/**
 * Box-Muller standard-normal sample. Called at init time only, so the trig
 * cost is irrelevant. This runs in the worker at runtime, where Math.random
 * is fine (workflow scripts can't use Math.random, but this isn't one).
 */
function randn(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** He initialization for ReLU layers: W ~ N(0, sqrt(2/fan_in)). */
export function heInit(out: number, fanIn: number): Float64Array {
  const std = Math.sqrt(2 / fanIn);
  const W = new Float64Array(out * fanIn);
  for (let i = 0; i < W.length; i++) W[i] = randn() * std;
  return W;
}

/** Small init for the linear output layer: W ~ N(0, 0.01). */
export function smallInit(out: number, fanIn: number): Float64Array {
  const W = new Float64Array(out * fanIn);
  for (let i = 0; i < W.length; i++) W[i] = randn() * 0.01;
  return W;
}
