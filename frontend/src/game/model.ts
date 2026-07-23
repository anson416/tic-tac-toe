// model.ts — pure Tic-Tac-Toe board logic, no DOM. Mirrors game.py's win
// detection (rows, columns, both diagonals) and X-first convention, but as
// immutable pure functions so transitions and UI snapshots can't be mutated
// out from under each other.

export type Cell = 0 | 1 | -1; // 0 = empty, 1 = X, -1 = O
export type Side = 1 | -1; // 1 = X (moves first), -1 = O

export const X: Side = 1;
export const O: Side = -1;

export interface BoardState {
  cells: Cell[]; // length 9
  turn: Side; // whose move it is
}

// The 8 winning lines, exactly matching game.py (rows, cols, diagonals).
export const WIN_LINES: ReadonlyArray<readonly [number, number, number]> = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
  [0, 3, 6], [1, 4, 7], [2, 5, 8], // cols
  [0, 4, 8], [2, 4, 6], // diagonals
];

export function newGame(): BoardState {
  return { cells: new Array(9).fill(0) as Cell[], turn: X };
}

export function cloneBoard(b: BoardState): BoardState {
  return { cells: b.cells.slice(), turn: b.turn };
}

/** Indices of empty cells — the legal moves for the side to move. */
export function legalMoves(b: BoardState): number[] {
  const out: number[] = [];
  for (let i = 0; i < 9; i++) if (b.cells[i] === 0) out.push(i);
  return out;
}

/** Returns a NEW BoardState with the move applied. Throws on illegal move. */
export function applyMove(b: BoardState, idx: number): BoardState {
  if (idx < 0 || idx > 8 || b.cells[idx] !== 0) {
    throw new Error(`illegal move at index ${idx}`);
  }
  const cells = b.cells.slice();
  cells[idx] = b.turn;
  return { cells, turn: b.turn === X ? O : X };
}

/** The winning Side, or null if no winner yet (a draw is null here — use isFull). */
export function winner(b: BoardState): Side | null {
  for (const [a, c, d] of WIN_LINES) {
    const v = b.cells[a];
    if (v !== 0 && v === b.cells[c] && v === b.cells[d]) return v as Side;
  }
  return null;
}

/** The specific winning line, if any — for the animation. */
export function winningLine(b: BoardState): readonly [number, number, number] | null {
  for (const line of WIN_LINES) {
    const [a, c, d] = line;
    const v = b.cells[a];
    if (v !== 0 && v === b.cells[c] && v === b.cells[d]) return line;
  }
  return null;
}

export function isFull(b: BoardState): boolean {
  for (let i = 0; i < 9; i++) if (b.cells[i] === 0) return false;
  return true;
}

export function isTerminal(b: BoardState): boolean {
  return winner(b) !== null || isFull(b);
}
