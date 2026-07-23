// board-controller.ts — renders the 3×3 board, handles clicks, and runs the
// CSS animations: piece placement, the winning-line draw, and the result
// flash (reqs 2, 11). Calls back to the GameController on a human click.
import { byId } from "../by-id";
import type { Cell, Side } from "../../game/model";
import { winningLine } from "../../game/model";

const SYMBOL_MARK: Record<1 | -1, string> = { 1: "×", [-1]: "○" };

export class BoardController {
  private cells: HTMLButtonElement[] = [];
  private winLine: SVGLineElement;
  private overlay: HTMLElement;
  private boardWrap: HTMLElement;
  private onCellClickCb: ((idx: number) => void) | null = null;
  private winSvg: SVGSVGElement;

  constructor() {
    this.winSvg = byId<SVGSVGElement>("overlay");
    this.winLine = byId<SVGLineElement>("win-line");
    this.overlay = byId("result-overlay");
    this.boardWrap = byId("board-wrap");
    for (let i = 0; i < 9; i++) {
      this.cells.push(byId<HTMLButtonElement>(`cell-${i}`));
    }
  }

  init(): void {
    for (let i = 0; i < 9; i++) {
      const idx = i;
      this.cells[i]!.addEventListener("click", () => this.onCellClickCb?.(idx));
    }
    byId("btn-new-game").addEventListener("click", () => this.onCellClickCb?.(-1));
    this.clearAnimations();
  }

  onCellClick(cb: (idx: number) => void): void {
    this.onCellClickCb = cb;
  }

  /** Render a board state. */
  render(cells: ReadonlyArray<Cell>, turn: Side, interactive: boolean): void {
    for (let i = 0; i < 9; i++) {
      const btn = this.cells[i]!;
      const v = cells[i];
      const mark = v === 0 ? "" : SYMBOL_MARK[v as 1 | -1];
      const prev = btn.textContent;
      btn.textContent = mark;
      btn.classList.toggle("x", v === 1);
      btn.classList.toggle("o", v === -1);
      // Animate newly-placed pieces (req 11).
      if (mark !== "" && prev === "") {
        btn.classList.remove("piece-in");
        // force reflow so the animation restarts
        void btn.offsetWidth;
        btn.classList.add("piece-in");
      }
      btn.disabled = !interactive || v !== 0;
    }
    // Turn is shown via the board's blue/red border, not text (req 2).
    this.boardWrap.classList.toggle("turn-x", turn === 1);
    this.boardWrap.classList.toggle("turn-o", turn === -1);
  }

  /** Draw the winning line over the 3 winning cells (req 11). */
  animateWin(cells: ReadonlyArray<Cell>): void {
    const line = winningLine({ cells: cells as Cell[], turn: 1 });
    if (!line) return;
    // Cell centers in the 300×300 viewBox (board is 3×3, cells ~100px each).
    const center = (idx: number) => ({
      x: (idx % 3) * 100 + 50,
      y: Math.floor(idx / 3) * 100 + 50,
    });
    const a = center(line[0]);
    const b = center(line[2]);
    this.winLine.setAttribute("x1", String(a.x));
    this.winLine.setAttribute("y1", String(a.y));
    this.winLine.setAttribute("x2", String(b.x));
    this.winLine.setAttribute("y2", String(b.y));
    const len = Math.hypot(b.x - a.x, b.y - a.y).toFixed(2);
    this.winLine.style.setProperty("--line-len", `${len}`);
    this.winLine.style.strokeDasharray = String(len);
    this.winLine.style.strokeDashoffset = String(len);
    this.winSvg.classList.remove("show");
    void this.winLine.getBoundingClientRect();
    this.winSvg.classList.add("show");
  }

  /** Show the result overlay ("X wins"/"O wins"/"Draw"). */
  showResult(winner: Side | null): void {
    this.overlay.textContent = winner === null ? "Draw" : `${winner === 1 ? "X" : "O"} wins`;
    this.overlay.hidden = false;
    this.overlay.classList.remove("flash");
    void this.overlay.offsetWidth;
    this.overlay.classList.add("flash");
    // Game over → no active turn, clear the turn border.
    this.boardWrap.classList.remove("turn-x", "turn-o");
  }

  clearAnimations(): void {
    this.winSvg.classList.remove("show");
    this.winLine.style.strokeDashoffset = String(0);
    this.overlay.hidden = true;
    this.overlay.classList.remove("flash");
    this.boardWrap.classList.remove("turn-x", "turn-o");
  }
}
