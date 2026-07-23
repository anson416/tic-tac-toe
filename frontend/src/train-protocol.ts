// train-protocol.ts — the message contract between the main thread and the
// training worker. Discriminated-by-shape (field-presence), id-keyed — no
// literal `type`/`kind` tag — mirroring the reference app's `"stripe" in r`
// pattern. Variant is told apart by which field it carries.
import type { Metrics, NetConfig } from "./nn/types";
export type { Metrics, NetConfig } from "./nn/types";

// ===== Requests (main → worker) =====
export interface StartTrainingReq {
  id: number;
  start: true;
}
export interface StopTrainingReq {
  id: number;
  stop: true;
}
export interface ReconfigureReq {
  id: number;
  config: NetConfig;
}
export interface InferenceReq {
  id: number;
  board: number[]; // length 9, -1/0/1
  side: number; // +1 X, -1 O
  smartness: number; // 0..1
}

export type WorkerReq =
  | StartTrainingReq
  | StopTrainingReq
  | ReconfigureReq
  | InferenceReq;

// ===== Responses (worker → main) =====
export interface MetricsResp {
  id: number;
  metrics: Metrics;
}
export interface InferenceResp {
  id: number;
  move: number; // 0..8
  activations: number[][]; // per-layer, for the heatmap
}
export interface ErrorResp {
  id: number;
  error: string;
}

export type WorkerResp = MetricsResp | InferenceResp | ErrorResp;

// ===== Discriminators (field presence, NOT a literal kind) =====
export function isStartReq(r: WorkerReq): r is StartTrainingReq {
  return "start" in r;
}
export function isStopReq(r: WorkerReq): r is StopTrainingReq {
  return "stop" in r;
}
export function isReconfigureReq(r: WorkerReq): r is ReconfigureReq {
  return "config" in r;
}
export function isInferenceReq(r: WorkerReq): r is InferenceReq {
  return "board" in r;
}
export function isMetrics(r: WorkerResp): r is MetricsResp {
  return "metrics" in r;
}
export function isInference(r: WorkerResp): r is InferenceResp {
  return "move" in r && "activations" in r;
}
export function isError(r: WorkerResp): r is ErrorResp {
  return "error" in r;
}

/** Default config used before the user applies their own. */
export function defaultConfig(): NetConfig {
  return {
    hiddenLayers: [{ neurons: 64 }, { neurons: 64 }],
    learningRate: 0.001,
    discount: 0.95,
    batchSize: 64,
    bufferSize: 20000,
    targetSyncEvery: 500,
    minEpsilon: 0.05,
    epsilonDecay: 0.9995,
  };
}
