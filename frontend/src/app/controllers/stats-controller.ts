// stats-controller.ts — X / O / Draws tallies with a Reset (req 9).
import { byId } from "../by-id";

export interface ScoreState {
  x: number;
  o: number;
  draw: number;
}

export class StatsController {
  private score: ScoreState = { x: 0, o: 0, draw: 0 };
  private elX: HTMLElement;
  private elO: HTMLElement;
  private elDraw: HTMLElement;

  constructor() {
    this.elX = byId("stat-x");
    this.elO = byId("stat-o");
    this.elDraw = byId("stat-draw");
  }

  init(): void {
    byId("btn-reset-stats").addEventListener("click", () => this.reset());
    this.render();
  }

  record(winner: 1 | -1 | null): void {
    if (winner === null) this.score.draw += 1;
    else if (winner === 1) this.score.x += 1;
    else this.score.o += 1;
    this.render();
  }

  reset(): void {
    this.score = { x: 0, o: 0, draw: 0 };
    this.render();
  }

  private render(): void {
    this.elX.textContent = String(this.score.x);
    this.elO.textContent = String(this.score.o);
    this.elDraw.textContent = String(this.score.draw);
  }
}
