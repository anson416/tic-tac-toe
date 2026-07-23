// network.ts — a small dense MLP: hidden layers ReLU, output layer LINEAR
// (raw Q-values, proper DQN). Holds weights, biases, Adam state, and exposes:
//   - forward(state) → { qValues, activations, cache }
//   - accumulateGrads(...)  — backprop ONE transition's TD error into buffers
//   - applyAdam(grads, lr)   — one Adam step from averaged (batch) gradients
//   - copyWeightsFrom(other) — target-net sync
// The worker's trainStep owns the batch loop: it zeroes grads, calls
// accumulateGrads per transition, divides by batch, then calls applyAdam once.
import { relu, reluDeriv } from "./activations";
import {
  adamStep,
  heInit,
  makeAdam,
  smallInit,
  type AdamState,
} from "./optimizer";
import { INPUT_SIZE, OUTPUT_SIZE } from "../game/encoding";

export interface DenseLayer {
  in: number;
  out: number;
  W: Float64Array; // row-major, shape out×in
  b: Float64Array; // length out
  relu: boolean; // true for hidden, false for linear output
}

export interface ForwardCache {
  z: number[][]; // pre-activations per layer (for backprop)
  a: number[][]; // activations per layer (a[0] = input, a[last] = output)
}

export interface GradBuffers {
  W: Float64Array[]; // one per layer, matching W shape
  b: Float64Array[]; // one per layer, matching b shape
}

export class Network {
  layers: DenseLayer[];
  adam: AdamState[];

  constructor(layerSpecs: ReadonlyArray<{ in: number; out: number; relu: boolean }>) {
    this.layers = layerSpecs.map((s) => ({
      in: s.in,
      out: s.out,
      relu: s.relu,
      W: s.relu ? heInit(s.out, s.in) : smallInit(s.out, s.in),
      b: new Float64Array(s.out),
    }));
    this.adam = this.layers.map((l) => makeAdam(l.W.length, l.b.length));
  }

  get outputSize(): number {
    return this.layers[this.layers.length - 1]!.out;
  }

  /** Forward pass. Returns Q-values, per-layer activations (for heatmap), and a backprop cache. */
  forward(state: ReadonlyArray<number>): { qValues: number[]; activations: number[][]; cache: ForwardCache } {
    const z: number[][] = [];
    const a: number[][] = [state.slice()];
    let cur = state;
    for (const layer of this.layers) {
      const zo = new Array<number>(layer.out);
      for (let o = 0; o < layer.out; o++) {
        let acc = layer.b[o];
        const row = o * layer.in;
        for (let i = 0; i < layer.in; i++) acc += layer.W[row + i] * cur[i]!;
        zo[o] = acc;
      }
      z.push(zo);
      const ao = layer.relu ? zo.map(relu) : zo.slice(); // output layer is linear
      a.push(ao);
      cur = ao;
    }
    return { qValues: a[a.length - 1]!.slice(), activations: a, cache: { z, a } };
  }

  /** Forward-only convenience (no cache) for fast inference. */
  qValues(state: ReadonlyArray<number>): number[] {
    let cur = state;
    for (const layer of this.layers) {
      const ao = new Array<number>(layer.out);
      for (let o = 0; o < layer.out; o++) {
        let acc = layer.b[o];
        const row = o * layer.in;
        for (let i = 0; i < layer.in; i++) acc += layer.W[row + i] * cur[i]!;
        ao[o] = layer.relu ? (acc > 0 ? acc : 0) : acc;
      }
      cur = ao;
    }
    return cur as number[];
  }

  /**
   * Backprop ONE transition's TD error into the provided grad buffers
   * (accumulates; caller zeroes first and averages + applies Adam after the
   * batch). Returns the per-sample TD loss (0.5 · delta²).
   *
   * The gradient touches ONLY Q[action]: δ at the output is zero except at
   * `action`, where it is (Q_online(s,a) − target).
   */
  accumulateGrads(
    cache: ForwardCache,
    action: number,
    target: number,
    grads: GradBuffers,
  ): number {
    const qValues = cache.a[cache.a.length - 1]!;
    const delta = qValues[action]! - target;

    const L = this.layers.length;
    const deltaOut = new Array<number>(this.layers[L - 1]!.out).fill(0);
    deltaOut[action] = delta;

    let deltaNext = deltaOut;
    for (let li = L - 1; li >= 0; li--) {
      const layer = this.layers[li]!;
      const aPrev = cache.a[li]!; // input to this layer
      const inN = layer.in;
      const gW = grads.W[li]!;
      const gB = grads.b[li]!;
      for (let o = 0; o < layer.out; o++) {
        const d = deltaNext[o]!;
        if (d === 0) continue;
        gB[o] += d;
        const row = o * inN;
        for (let i = 0; i < inN; i++) gW[row + i] += d * aPrev[i]!;
      }
      if (li > 0) {
        const prevOut = this.layers[li - 1]!.out;
        const newDelta = new Array<number>(prevOut).fill(0);
        for (let i = 0; i < prevOut; i++) {
          let acc = 0;
          for (let o = 0; o < layer.out; o++) acc += layer.W[o * inN + i]! * deltaNext[o]!;
          newDelta[i] = this.layers[li - 1]!.relu ? acc * reluDeriv(cache.z[li - 1]![i]!) : acc;
        }
        deltaNext = newDelta;
      }
    }
    return 0.5 * delta * delta;
  }

  /** Divide grads by batch size, then one Adam step per layer. Resets grads to zero. */
  applyAdam(grads: GradBuffers, lr: number, batchSize: number): void {
    for (let li = 0; li < this.layers.length; li++) {
      const layer = this.layers[li]!;
      const gW = grads.W[li]!;
      const gB = grads.b[li]!;
      for (let i = 0; i < gW.length; i++) gW[i] /= batchSize;
      for (let i = 0; i < gB.length; i++) gB[i] /= batchSize;
      adamStep(this.adam[li]!, layer.W, layer.b, gW, gB, lr);
      gW.fill(0);
      gB.fill(0);
    }
  }

  /** Allocate zeroed grad buffers matching this network's shape. */
  newGradBuffers(): GradBuffers {
    return {
      W: this.layers.map((l) => new Float64Array(l.W.length)),
      b: this.layers.map((l) => new Float64Array(l.b.length)),
    };
  }

  /** Deep-copy weights/biases from another network (same architecture). */
  copyWeightsFrom(other: Network): void {
    for (let li = 0; li < this.layers.length; li++) {
      this.layers[li]!.W.set(other.layers[li]!.W);
      this.layers[li]!.b.set(other.layers[li]!.b);
    }
  }
}

/** Build a network from hidden-layer sizes: [INPUT] -> hidden(ReLU)* -> [OUTPUT linear]. */
export function buildNetwork(hiddenLayerSizes: ReadonlyArray<number>): Network {
  const sizes = [INPUT_SIZE, ...hiddenLayerSizes, OUTPUT_SIZE];
  const specs: { in: number; out: number; relu: boolean }[] = [];
  for (let i = 0; i < sizes.length - 1; i++) {
    specs.push({ in: sizes[i]!, out: sizes[i + 1]!, relu: i < sizes.length - 2 });
  }
  return new Network(specs);
}
