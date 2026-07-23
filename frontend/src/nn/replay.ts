// replay.ts — ring-buffer experience replay. Stores transitions from both
// X-seats and O-seats (each carries its sideFlag so replay re-encodes
// consistently). Overwrites oldest when full; samples random mini-batches
// as soon as at least `batchSize` transitions exist (no warmup wait).
import type { Transition } from "./types";

export class ReplayBuffer {
  private buf: Transition[];
  private cap: number;
  private head = 0; // next write index
  private n = 0; // number stored

  constructor(capacity: number) {
    this.cap = Math.max(1, Math.floor(capacity));
    this.buf = new Array<Transition>(this.cap);
  }

  get size(): number {
    return this.n;
  }

  push(t: Transition): void {
    this.buf[this.head] = t;
    this.head = (this.head + 1) % this.cap;
    if (this.n < this.cap) this.n += 1;
  }

  /** Random sample of up to `count` transitions (fewer if the buffer is smaller). */
  sample(count: number): Transition[] {
    const k = Math.min(count, this.n);
    const out: Transition[] = [];
    for (let i = 0; i < k; i++) {
      const idx = (Math.random() * this.n) | 0;
      out.push(this.buf[idx]!);
    }
    return out;
  }

  clear(): void {
    this.buf = new Array<Transition>(this.cap);
    this.head = 0;
    this.n = 0;
  }

  resize(capacity: number): void {
    this.cap = Math.max(1, Math.floor(capacity));
    this.buf = new Array<Transition>(this.cap);
    this.head = 0;
    this.n = 0;
  }
}
