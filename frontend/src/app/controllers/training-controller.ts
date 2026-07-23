// training-controller.ts — Start/Stop toggle + live games/steps/ε counters
// (reqs 1, 7). Subscribes to worker metrics pushes and reflects them.
import { byId } from "../by-id";
import type { WorkerBackendController } from "../worker-client";
import type { Metrics } from "../../nn/types";

export class TrainingController {
  private running = false;
  private elToggle: HTMLButtonElement;
  private elGames: HTMLElement;
  private elSteps: HTMLElement;
  private elEpsilon: HTMLElement;

  constructor(private worker: WorkerBackendController) {
    this.elToggle = byId<HTMLButtonElement>("btn-train-toggle");
    this.elGames = byId("games-played");
    this.elSteps = byId("steps-count");
    this.elEpsilon = byId("epsilon-value");
  }

  init(): void {
    this.elToggle.addEventListener("click", () => this.toggle());
    this.worker.onMetrics((m) => this.onMetrics(m));
    this.render();
  }

  toggle(): void {
    this.running = !this.running;
    if (this.running) {
      this.worker.control({ start: true });
    } else {
      this.worker.control({ stop: true });
    }
    this.render();
  }

  get isRunning(): boolean {
    return this.running;
  }

  private onMetrics(m: Metrics): void {
    this.elGames.textContent = String(m.gamesPlayed);
    this.elSteps.textContent = String(m.stepCount);
    this.elEpsilon.textContent = m.epsilon.toFixed(3);
  }

  private render(): void {
    this.elToggle.textContent = this.running ? "Stop training" : "Start training";
    this.elToggle.classList.toggle("running", this.running);
  }
}
