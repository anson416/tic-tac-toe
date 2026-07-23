// game-controller.ts — the orchestrator. Owns the board state machine
// (idle / x-turn / o-turn / ai-thinking / game-over), interleaves human
// clicks and AI moves, applies the AI think-delay, and guards against stale
// inference responses. The AI-vs-AI case chains moves via setTimeout so the
// page never freezes and animations render (reqs 1, 3, 7, 8, 11).
import type { WorkerBackendController } from "./worker-client";
import {
  applyMove,
  isTerminal,
  legalMoves,
  newGame,
  winner,
  type BoardState,
  type Side,
  X,
} from "../game/model";
import type { BoardController } from "./controllers/board-controller";
import type { PlayersController } from "./controllers/players-controller";
import type { StatsController } from "./controllers/stats-controller";
import type { HeatmapController } from "./controllers/heatmap-controller";

type Phase = "idle" | "x-turn" | "o-turn" | "ai-thinking" | "game-over";

const THINK_MIN_MS = 400;
const THINK_MAX_MS = 900;

function randDelay(): number {
  return THINK_MIN_MS + Math.random() * (THINK_MAX_MS - THINK_MIN_MS);
}

export class GameController {
  private board: BoardState = newGame();
  private phase: Phase = "idle";
  private currentInferenceId = 0;
  private smartness = 1; // 0..1, 1 = greedy
  private scheduled = 0; // timeout id for AI chaining / think-delay

  constructor(
    private worker: WorkerBackendController,
    private boardCtl: BoardController,
    private playersCtl: PlayersController,
    private statsCtl: StatsController,
    private heatmapCtl: HeatmapController,
  ) {}

  init(): void {
    this.boardCtl.onCellClick((idx) => this.onUserAction(idx));
    this.playersCtl.onChange(() => this.onSeatsChanged());
    this.startNewGame();
  }

  setSmartness(s: number): void {
    this.smartness = s;
  }

  // --- flow ----------------------------------------------------------------

  private onUserAction(idx: number): void {
    // idx === -1 → New game button.
    if (idx === -1) {
      this.startNewGame();
      return;
    }
    if (this.phase === "game-over") return;
    const side = this.board.turn;
    const seat = this.playersCtl.seatForSide(side);
    if (seat !== "human") return; // not a human's turn
    if (!legalMoves(this.board).includes(idx)) return;
    this.applyMoveAndAdvance(idx, null);
  }

  private onSeatsChanged(): void {
    // Restart so the new seat assignment takes effect immediately.
    this.startNewGame();
  }

  private startNewGame(): void {
    if (this.scheduled) {
      clearTimeout(this.scheduled);
      this.scheduled = 0;
    }
    this.currentInferenceId++; // invalidate any in-flight AI move
    this.board = newGame();
    this.boardCtl.clearAnimations();
    this.advance();
  }

  private advance(): void {
    if (isTerminal(this.board)) {
      this.endGame();
      return;
    }
    const side = this.board.turn;
    const seat = this.playersCtl.seatForSide(side);
    if (seat === "human") {
      this.phase = side === X ? "x-turn" : "o-turn";
      this.boardCtl.render(this.board.cells, side, true);
    } else {
      this.phase = "ai-thinking";
      this.boardCtl.render(this.board.cells, side, false);
      this.doAiMove(side);
    }
  }

  private endGame(): void {
    this.phase = "game-over";
    const w = winner(this.board);
    this.statsCtl.record(w);
    this.boardCtl.render(this.board.cells, this.board.turn, false);
    if (w !== null) this.boardCtl.animateWin(this.board.cells);
    this.boardCtl.showResult(w);
  }

  private applyMoveAndAdvance(
    idx: number,
    activations: number[][] | null,
    aiSide?: Side,
  ): void {
    this.board = applyMove(this.board, idx);
    // Only refresh the heatmap for AI moves, where we have both the network
    // activations and the AI's side (needed to color the output row).
    if (activations && aiSide !== undefined) {
      this.heatmapCtl.update(activations, aiSide);
    }
    this.boardCtl.render(this.board.cells, this.board.turn, false);
    this.advance();
  }

  // --- AI move with think-delay + anti-race (reqs 8, 1) --------------------

  private async doAiMove(side: Side): Promise<void> {
    const reqId = ++this.currentInferenceId;
    // Actual inference + a random floor delay. Inference is fast; the floor
    // gives the "thinking" feel. Both AI-AI and human-AI use this.
    try {
      const result = await this.worker.requestInference({
        board: this.board.cells.slice(),
        side,
        smartness: this.smartness,
      });
      if (reqId !== this.currentInferenceId) return; // stale
      const inferenceMs = 0; // inference already elapsed inside the await
      const remaining = Math.max(0, randDelay() - inferenceMs);
      await this.sleep(remaining);
      if (reqId !== this.currentInferenceId) return; // stale after delay
      const move = result.move;
      if (!legalMoves(this.board).includes(move)) {
        // Defensive fallback: pick a random legal move (shouldn't happen).
        const lm = legalMoves(this.board);
        if (lm.length === 0) return;
        const fallback = lm[(Math.random() * lm.length) | 0];
        this.applyMoveAndAdvance(fallback, result.activations, side);
        return;
      }
      this.applyMoveAndAdvance(move, result.activations, side);
    } catch {
      // Ignore transient worker errors; the game stays in ai-thinking but
      // the user can press New game to recover.
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => {
      this.scheduled = window.setTimeout(resolve, ms) as unknown as number;
    });
  }
}
