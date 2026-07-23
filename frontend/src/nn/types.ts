// types.ts — shared shapes for the neural net + worker protocol.
import type { Side } from "../game/model";

export interface LayerSpec {
  neurons: number;
}

export interface NetConfig {
  hiddenLayers: LayerSpec[]; // e.g. [{neurons:64},{neurons:64}]
  learningRate: number; // Adam lr
  discount: number; // γ
  batchSize: number;
  bufferSize: number;
  targetSyncEvery: number;
  minEpsilon: number;
  epsilonDecay: number;
}

/** Per-layer activations, index 0 = input, last = output Q-values. */
export interface ActivationSnapshot {
  layers: number[][];
}

export interface Transition {
  state: number[]; // 10-vector
  action: number; // 0..8 (always legal)
  reward: number; // terminal reward at game end; 0 non-terminal
  nextState: number[]; // 10-vector; equals state when done
  done: boolean;
  sideFlag: Side; // +1 X / -1 O — stored so replay re-encodes consistently
}

export interface Metrics {
  gamesPlayed: number;
  stepCount: number;
  epsilon: number;
  smoothedLoss: number;
  winRateVsRandom: number; // 0..1
}
