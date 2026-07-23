// main.ts — bootstrap. Wires the dependency graph in order: worker client →
// game controller → each controller .init(). Sets the status indicator to
// "ready" (the smoke test polls this). Fire-and-forget per the reference.
import "./style.css";
import { byId } from "./app/by-id";
import { WorkerBackendController } from "./app/worker-client";
import { GameController } from "./app/game-controller";
import { BoardController } from "./app/controllers/board-controller";
import { PlayersController } from "./app/controllers/players-controller";
import { ModelConfigController } from "./app/controllers/model-config-controller";
import { TrainingController } from "./app/controllers/training-controller";
import { StatsController } from "./app/controllers/stats-controller";
import { HeatmapController } from "./app/controllers/heatmap-controller";
import { ChartsController } from "./app/controllers/charts-controller";
import { toggleTheme } from "./app/ui/theme";
import type { Metrics } from "./nn/types";

function setStatus(cls: string, text: string): void {
  const el = byId("app-status");
  el.className = `status ${cls}`;
  el.textContent = text;
}

async function main(): Promise<void> {
  setStatus("status-booting", "booting…");

  const worker = new WorkerBackendController();
  await worker.init();

  const board = new BoardController();
  const players = new PlayersController();
  const stats = new StatsController();
  const heatmap = new HeatmapController();
  const charts = new ChartsController();
  const training = new TrainingController(worker);
  const modelCfg = new ModelConfigController(worker, charts);

  board.init();
  players.init();
  stats.init();
  heatmap.init();
  charts.init();
  training.init();
  modelCfg.init();

  // Route metrics pushes to the charts panel.
  worker.onMetrics((m: Metrics) => charts.onMetrics(m));

  const game = new GameController(worker, board, players, stats, heatmap);
  game.init();

  // Smartness slider → inference temperature (req 10).
  const slider = byId<HTMLInputElement>("smartness-slider");
  const smartLabel = byId("smartness-value");
  const updateSmartness = (): void => {
    const pct = slider.valueAsNumber;
    smartLabel.textContent = `${pct}%`;
    game.setSmartness(pct / 100);
  };
  slider.addEventListener("input", updateSmartness);
  updateSmartness();

  // Theme toggle.
  byId("btn-theme").addEventListener("click", () => toggleTheme());

  setStatus("status-ready", "ready");
}

main().catch((err) => {
  console.error(err);
  setStatus("status-error", `error: ${err instanceof Error ? err.message : String(err)}`);
});
