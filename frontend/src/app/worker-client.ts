// worker-client.ts — main-thread handle on the training worker. Spawns it,
// pairs requests to responses by id, and routes unsolicited MetricsResp
// (the worker pushes these during training) to a metrics listener.
import {
  isMetrics,
  type InferenceReq,
  type InferenceResp,
  type Metrics,
  type WorkerReq,
  type WorkerResp,
} from "../train-protocol";

export type MetricsListener = (m: Metrics) => void;

export class WorkerBackendController {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (r: WorkerResp) => void; reject: (e: unknown) => void }
  >();
  private metricsListeners = new Set<MetricsListener>();

  constructor() {
    this.worker = new Worker(new URL("../workers/train.worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (e: MessageEvent<WorkerResp>) => this.onMessage(e.data);
    this.worker.onerror = (ev: ErrorEvent) => {
      const err = new Error(ev.message || "train worker error");
      for (const { reject } of this.pending.values()) reject(err);
      this.pending.clear();
    };
  }

  init(): Promise<void> {
    // Give the worker a tick to be ready. No handshake message is strictly
    // required since the first request will simply queue.
    return Promise.resolve();
  }

  private onMessage(data: WorkerResp): void {
    if (isMetrics(data)) {
      // Unsolicited metrics push during training (id === 0).
      for (const fn of this.metricsListeners) fn(data.metrics);
      return;
    }
    const p = this.pending.get(data.id);
    if (!p) return;
    this.pending.delete(data.id);
    p.resolve(data);
  }

  onMetrics(fn: MetricsListener): () => void {
    this.metricsListeners.add(fn);
    return () => this.metricsListeners.delete(fn);
  }

  /** Send a request expecting a matching-id response. */
  request<T extends WorkerResp>(req: WorkerReq): Promise<T> {
    const id = this.nextId++;
    const payload = { ...req, id } as WorkerReq;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (r) => resolve(r as T),
        reject,
      });
      this.worker.postMessage(payload);
    });
  }

  /** Fire-and-forget control messages (start/stop/reconfigure). Assigns the id. */
  control(req: Omit<WorkerReq, "id">): void {
    const id = this.nextId++;
    this.worker.postMessage({ ...req, id } as WorkerReq);
  }

  /** Inference request → resolved move + activations. */
  requestInference(req: Omit<InferenceReq, "id">): Promise<InferenceResp> {
    return this.request<InferenceResp>({ ...req } as InferenceReq);
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
  }
}
